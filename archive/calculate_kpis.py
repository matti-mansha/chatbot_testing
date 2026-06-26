#!/usr/bin/env python3
"""
calculate_kpis.py
KPI Calculator for the Chatbot Testing System.

Fetches evaluation data from Notion, calculates aggregate KPIs across
test runs, and outputs a formatted report. Optionally writes summary
KPIs back to the Test Runs database.

Usage:
    python calculate_kpis.py                     # All completed test runs
    python calculate_kpis.py --run TR1           # Specific test run
    python calculate_kpis.py --run TR1 --json    # Output raw JSON
    python calculate_kpis.py --run TR1 --write   # Write summary back to Notion
    python calculate_kpis.py --compare TR1 TR2   # Compare two runs
"""
import os
import sys
import json
import time
import argparse
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import httpx
from dotenv import load_dotenv

from logging_config import setup_logging, log_exception, log_api_call

# =====================================
# SETUP
# =====================================

load_dotenv()
logger = setup_logging("calculate_kpis")

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
TEST_CASE_EXECUTIONS_PAGE_ID = os.getenv("TEST_CASE_EXECUTIONS_DB_ID")
TEST_RUNS_PAGE_ID = os.getenv("TEST_RUNS_DB_ID")
EVALUATION_PAGES_PARENT_PAGE_ID = os.getenv("EVALUATION_PAGES_DB_ID")

NOTION_API_BASE = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

# The 9 KPI names matching the evaluation output
KPI_NAMES = [
    "task_completeness",
    "user_comfort",
    "understanding_and_relevance",
    "clarity_and_actionability",
    "edge_cases_and_constraints",
    "proactiveness_and_guidance",
    "tone_and_personalization",
    "accuracy_and_policy_compliance",
    "efficiency_and_flow",
]

KPI_DISPLAY = {k: k.replace("_", " ").title() for k in KPI_NAMES}


# =====================================
# NOTION HELPERS
# =====================================

def format_notion_id(notion_id: str) -> str:
    clean = (notion_id or "").replace("-", "")
    if len(clean) == 32:
        return f"{clean[0:8]}-{clean[8:12]}-{clean[12:16]}-{clean[16:20]}-{clean[20:32]}"
    return notion_id


def find_database_in_page(page_id: str) -> Optional[str]:
    if not page_id:
        return None
    page_uuid = format_notion_id(page_id)
    try:
        resp = httpx.get(
            f"{NOTION_API_BASE}/blocks/{page_uuid}/children",
            headers=HEADERS,
            timeout=30.0,
        )
        resp.raise_for_status()
        for block in resp.json().get("results", []):
            if block.get("type") in ["child_database", "table"]:
                return block["id"]
        return page_uuid
    except Exception:
        return page_uuid


def extract_property_value(page: Dict[str, Any], property_name: str) -> Any:
    try:
        prop = page["properties"].get(property_name, {})
        ptype = prop.get("type")
        if ptype == "title":
            lst = prop.get("title", [])
            return lst[0]["plain_text"] if lst else ""
        elif ptype == "rich_text":
            lst = prop.get("rich_text", [])
            return "".join(item.get("plain_text", "") for item in lst) if lst else ""
        elif ptype == "number":
            return prop.get("number")
        elif ptype == "status":
            st = prop.get("status")
            return st["name"] if st else None
        elif ptype == "date":
            d = prop.get("date")
            return d["start"] if d else None
        else:
            return None
    except Exception:
        return None


def extract_rich_text_link(page: Dict[str, Any], property_name: str) -> Optional[str]:
    try:
        prop = page["properties"].get(property_name, {})
        if prop.get("type") != "rich_text":
            return None
        lst = prop.get("rich_text", [])
        if not lst:
            return None
        first = lst[0]
        href = first.get("href")
        if href:
            return href
        text = first.get("text", {})
        link = text.get("link")
        if link and "url" in link:
            return link["url"]
        return None
    except Exception:
        return None


def extract_page_id_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    base = url.split("?", 1)[0].split("#", 1)[0]
    parts = base.rstrip("/").split("/")
    if not parts:
        return None
    last = parts[-1]
    hex_only = "".join(re.findall(r"[0-9a-fA-F]", last))
    if len(hex_only) < 32:
        return None
    clean = hex_only[-32:]
    return format_notion_id(clean)


def notion_query_all(db_id: str, filter_body: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Query a Notion database with pagination, returning all results."""
    results = []
    cursor = None
    while True:
        body: Dict[str, Any] = {}
        if filter_body:
            body["filter"] = filter_body
        if cursor:
            body["start_cursor"] = cursor

        resp = httpx.post(
            f"{NOTION_API_BASE}/databases/{db_id}/query",
            headers=HEADERS,
            json=body,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        results.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")

    return results


# =====================================
# EVALUATION DATA FETCHERS
# =====================================

def fetch_evaluation_json_from_page(page_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch evaluation JSON from an evaluation page.
    The raw JSON is split across multiple consecutive code blocks (Notion's
    2000-char limit per block), so we concatenate all code blocks after the
    "Raw evaluation JSON" heading and parse the combined text.
    """
    page_uuid = format_notion_id(page_id)
    try:
        # Fetch all blocks from the page
        all_blocks = []
        cursor = None
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            resp = httpx.get(
                f"{NOTION_API_BASE}/blocks/{page_uuid}/children",
                headers=HEADERS,
                params=params,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            all_blocks.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

        # Find the "Raw evaluation JSON" heading, then concatenate all
        # consecutive code blocks that follow it
        found_heading = False
        code_chunks = []

        for block in all_blocks:
            btype = block.get("type", "")

            # Detect the heading
            if not found_heading and "heading" in btype:
                rich_text = block.get(btype, {}).get("rich_text", [])
                text = "".join(rt.get("plain_text", "") for rt in rich_text)
                if "raw evaluation json" in text.lower():
                    found_heading = True
                    continue

            # After the heading, collect code blocks
            if found_heading:
                if btype == "code":
                    rich_text = block.get("code", {}).get("rich_text", [])
                    code_chunks.append("".join(rt.get("plain_text", "") for rt in rich_text))
                else:
                    # Stop at the first non-code block after heading
                    break

        if code_chunks:
            combined = "".join(code_chunks)
            try:
                return json.loads(combined)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse concatenated JSON ({len(combined)} chars) from page {page_id}")

        # Fallback: try any code block that starts with '{'
        for block in all_blocks:
            if block.get("type") == "code":
                rich_text = block.get("code", {}).get("rich_text", [])
                code_text = "".join(rt.get("plain_text", "") for rt in rich_text)
                if code_text.strip().startswith("{"):
                    try:
                        return json.loads(code_text)
                    except json.JSONDecodeError:
                        continue

        return None
    except Exception as e:
        logger.warning(f"Could not fetch evaluation JSON from page {page_id}: {e}")
        return None


def get_evaluated_executions(db_id: str, test_run_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch all executions with status 'Evaluation completed'.
    Optionally filter by test run prefix (e.g., 'TR1').
    """
    logger.info(f"Fetching evaluated executions" + (f" for run {test_run_prefix}" if test_run_prefix else ""))

    results = notion_query_all(db_id, {
        "property": "Test Execution Status",
        "status": {"equals": "Evaluation completed"},
    })

    if test_run_prefix:
        results = [
            r for r in results
            if (extract_property_value(r, "Test run number") or "").startswith(test_run_prefix + ".")
               or extract_property_value(r, "Test run number") == test_run_prefix
        ]

    logger.info(f"Found {len(results)} evaluated executions")
    return results


# =====================================
# DATA COLLECTION
# =====================================

def collect_execution_data(executions: List[Dict[str, Any]], fetch_kpi_details: bool = True) -> List[Dict[str, Any]]:
    """
    Collect structured data from executions, optionally fetching
    detailed KPI scores from evaluation pages.
    """
    data = []
    total = len(executions)

    for i, execution in enumerate(executions, 1):
        run_number = extract_property_value(execution, "Test run number") or ""
        test_case_name = extract_property_value(execution, "Test case name") or ""
        persona = extract_property_value(execution, "Persona") or ""
        overall_score = extract_property_value(execution, "Evaluation Score")
        num_turns = extract_property_value(execution, "Number of Turns")
        duration = extract_property_value(execution, "Execution duration")
        exec_date = extract_property_value(execution, "Execution date / time")

        # Extract test run prefix (e.g., "TR1" from "TR1.TC60.5")
        run_prefix = run_number.split(".")[0] if "." in run_number else run_number

        entry = {
            "run_number": run_number,
            "run_prefix": run_prefix,
            "test_case_name": test_case_name,
            "persona": persona,
            "overall_score": overall_score,
            "num_turns": num_turns,
            "duration": duration,
            "exec_date": exec_date,
            "kpis": {},
            "result": None,
        }

        # Fetch detailed KPIs from evaluation page
        if fetch_kpi_details:
            eval_link = extract_rich_text_link(execution, "Evaluation JSON")
            if eval_link:
                eval_page_id = extract_page_id_from_url(eval_link)
                if eval_page_id:
                    print(f"  [{i}/{total}] Fetching KPIs for {run_number} ({test_case_name[:40]})...", end="\r", file=sys.stderr)
                    eval_json = fetch_evaluation_json_from_page(eval_page_id)
                    if eval_json:
                        kpis = eval_json.get("kpis", {})
                        for kpi_name in KPI_NAMES:
                            kpi_data = kpis.get(kpi_name, {})
                            entry["kpis"][kpi_name] = kpi_data.get("score")

                        overall = eval_json.get("overall", {})
                        entry["result"] = overall.get("result")

                        # Use more precise overall score from eval JSON
                        json_score = overall.get("overall_score")
                        if json_score is not None:
                            entry["overall_score"] = json_score
                    time.sleep(0.2)  # Rate limiting

        data.append(entry)

    print()  # Clear the progress line
    return data


# =====================================
# KPI CALCULATIONS
# =====================================

def calc_pass_rate(data: List[Dict]) -> Dict[str, Any]:
    """Overall Pass Rate: % of executions with result = PASS"""
    results = [d["result"] for d in data if d["result"]]
    if not results:
        return {"value": None, "detail": "No results available"}

    total = len(results)
    passed = sum(1 for r in results if r == "PASS")
    partial = sum(1 for r in results if r == "PARTIAL")
    failed = sum(1 for r in results if r == "FAIL")

    return {
        "value": round(passed / total * 100, 1) if total else 0,
        "unit": "%",
        "detail": f"{passed} PASS / {partial} PARTIAL / {failed} FAIL (of {total})",
    }


def calc_avg_score(data: List[Dict]) -> Dict[str, Any]:
    """Average Evaluation Score"""
    scores = [d["overall_score"] for d in data if d["overall_score"] is not None]
    if not scores:
        return {"value": None, "detail": "No scores available"}

    avg = sum(scores) / len(scores)
    return {
        "value": round(avg, 1),
        "unit": "/100",
        "detail": f"Range: {min(scores)}-{max(scores)}, Median: {sorted(scores)[len(scores)//2]}",
    }


def calc_score_distribution(data: List[Dict]) -> Dict[str, Any]:
    """Score Distribution across quality bands"""
    scores = [d["overall_score"] for d in data if d["overall_score"] is not None]
    if not scores:
        return {"value": None, "detail": "No scores available"}

    total = len(scores)
    bands = {
        "Excellent (81-100)": sum(1 for s in scores if s >= 81),
        "Good (61-80)": sum(1 for s in scores if 61 <= s <= 80),
        "Acceptable (41-60)": sum(1 for s in scores if 41 <= s <= 60),
        "Poor (1-40)": sum(1 for s in scores if s <= 40),
    }

    return {
        "value": {k: f"{v}/{total} ({round(v/total*100)}%)" for k, v in bands.items()},
        "detail": f"Total: {total} evaluations",
    }


def calc_weakest_kpi(data: List[Dict]) -> Dict[str, Any]:
    """Weakest KPI: which of the 9 KPIs has the lowest average"""
    kpi_scores = defaultdict(list)
    for d in data:
        for kpi_name, score in d.get("kpis", {}).items():
            if score is not None:
                kpi_scores[kpi_name].append(score)

    if not kpi_scores:
        return {"value": None, "detail": "No KPI data available (run with --detailed)"}

    kpi_avgs = {k: round(sum(v) / len(v), 1) for k, v in kpi_scores.items() if v}
    if not kpi_avgs:
        return {"value": None, "detail": "No KPI averages"}

    sorted_kpis = sorted(kpi_avgs.items(), key=lambda x: x[1])
    weakest = sorted_kpis[0]
    strongest = sorted_kpis[-1]

    return {
        "value": f"{KPI_DISPLAY.get(weakest[0], weakest[0])}: {weakest[1]}/100",
        "all_averages": {KPI_DISPLAY.get(k, k): v for k, v in sorted_kpis},
        "detail": f"Strongest: {KPI_DISPLAY.get(strongest[0], strongest[0])} ({strongest[1]})",
    }


def calc_safety_compliance(data: List[Dict]) -> Dict[str, Any]:
    """
    Safety Compliance Rate: for safety-related test cases,
    % where accuracy_and_policy_compliance >= 41
    """
    safety_keywords = ["safety", "abuse", "emergency", "violence", "threat", "self-harm",
                       "exploit", "danger", "confiscated", "assault", "harass"]

    safety_executions = []
    for d in data:
        name_lower = (d.get("test_case_name") or "").lower()
        if any(kw in name_lower for kw in safety_keywords):
            safety_executions.append(d)

    if not safety_executions:
        return {"value": "N/A", "detail": "No safety-related test cases found in this run"}

    compliant = 0
    total = 0
    for d in safety_executions:
        acc_score = d.get("kpis", {}).get("accuracy_and_policy_compliance")
        if acc_score is not None:
            total += 1
            if acc_score >= 41:
                compliant += 1

    if total == 0:
        return {"value": "N/A", "detail": f"{len(safety_executions)} safety cases but no KPI data"}

    return {
        "value": round(compliant / total * 100, 1),
        "unit": "%",
        "detail": f"{compliant}/{total} safety test cases with accuracy_and_policy_compliance >= 41",
    }


def calc_workflow_compliance(data: List[Dict]) -> Dict[str, Any]:
    """
    Support Workflow Compliance Rate: for workflow test cases,
    based on overall result and key KPIs.
    """
    workflow_keywords = ["support", "workflow", "contact form", "escalation", "webform", "freshdesk"]

    workflow_executions = []
    for d in data:
        name_lower = (d.get("test_case_name") or "").lower()
        if any(kw in name_lower for kw in workflow_keywords):
            workflow_executions.append(d)

    if not workflow_executions:
        return {"value": "N/A", "detail": "No workflow-related test cases found"}

    passed = sum(1 for d in workflow_executions if d.get("result") == "PASS")
    total = len(workflow_executions)

    return {
        "value": round(passed / total * 100, 1) if total else 0,
        "unit": "%",
        "detail": f"{passed}/{total} workflow test cases passed",
    }


def calc_avg_turns(data: List[Dict]) -> Dict[str, Any]:
    """Average Turns to Resolution"""
    turns = [d["num_turns"] for d in data if d["num_turns"] is not None]
    if not turns:
        return {"value": None, "detail": "No turn data available"}

    avg = sum(turns) / len(turns)
    over_10 = sum(1 for t in turns if t > 10)

    return {
        "value": round(avg, 1),
        "unit": "turns",
        "detail": f"Range: {min(turns)}-{max(turns)}, {over_10}/{len(turns)} exceeded 10 turns",
    }


def calc_avg_duration(data: List[Dict]) -> Dict[str, Any]:
    """Average Conversation Duration"""
    durations = [d["duration"] for d in data if d["duration"] is not None]
    if not durations:
        return {"value": None, "detail": "No duration data available"}

    avg = sum(durations) / len(durations)
    return {
        "value": round(avg, 1),
        "unit": "seconds",
        "detail": f"Range: {round(min(durations),1)}s-{round(max(durations),1)}s, "
                  f"Total: {round(sum(durations)/60,1)} min",
    }


def calc_overstaying_rate(data: List[Dict]) -> Dict[str, Any]:
    """Overstaying Rate: % of conversations exceeding 10 turns"""
    turns = [d["num_turns"] for d in data if d["num_turns"] is not None]
    if not turns:
        return {"value": None, "detail": "No turn data available"}

    over_10 = sum(1 for t in turns if t > 10)
    total = len(turns)

    return {
        "value": round(over_10 / total * 100, 1) if total else 0,
        "unit": "%",
        "detail": f"{over_10}/{total} conversations exceeded 10 turns",
    }


def calc_persona_fairness(data: List[Dict]) -> Dict[str, Any]:
    """Persona Fairness Gap: score difference between persona groups"""
    persona_scores = defaultdict(list)
    for d in data:
        persona = (d.get("persona") or "").strip()
        score = d.get("overall_score")
        if persona and score is not None:
            # Normalize persona to a group (au pair vs host family)
            persona_lower = persona.lower()
            if any(kw in persona_lower for kw in ["host", "family", "gastfamilie"]):
                group = "Host Family"
            elif any(kw in persona_lower for kw in ["au pair", "aupair", "au-pair"]):
                group = "Au Pair"
            else:
                group = "Other"
            persona_scores[group].append(score)

    if len(persona_scores) < 2:
        return {"value": "N/A", "detail": "Need at least 2 persona groups to compare"}

    group_avgs = {g: round(sum(s) / len(s), 1) for g, s in persona_scores.items() if s}
    if len(group_avgs) < 2:
        return {"value": "N/A", "detail": "Not enough scored groups"}

    scores_list = list(group_avgs.values())
    gap = max(scores_list) - min(scores_list)

    return {
        "value": round(gap, 1),
        "unit": "points",
        "groups": group_avgs,
        "detail": " | ".join(f"{g}: {s}" for g, s in sorted(group_avgs.items())),
    }


def calc_worst_test_cases(data: List[Dict], n: int = 5) -> Dict[str, Any]:
    """Worst N Test Cases by average overall score"""
    tc_scores = defaultdict(list)
    for d in data:
        name = d.get("test_case_name")
        score = d.get("overall_score")
        if name and score is not None:
            tc_scores[name].append(score)

    if not tc_scores:
        return {"value": None, "detail": "No data"}

    tc_avgs = {k: round(sum(v) / len(v), 1) for k, v in tc_scores.items()}
    sorted_tcs = sorted(tc_avgs.items(), key=lambda x: x[1])[:n]

    return {
        "value": sorted_tcs,
        "detail": f"Bottom {n} of {len(tc_avgs)} unique test cases",
    }


def calc_best_test_cases(data: List[Dict], n: int = 5) -> Dict[str, Any]:
    """Best N Test Cases by average overall score"""
    tc_scores = defaultdict(list)
    for d in data:
        name = d.get("test_case_name")
        score = d.get("overall_score")
        if name and score is not None:
            tc_scores[name].append(score)

    if not tc_scores:
        return {"value": None, "detail": "No data"}

    tc_avgs = {k: round(sum(v) / len(v), 1) for k, v in tc_scores.items()}
    sorted_tcs = sorted(tc_avgs.items(), key=lambda x: x[1], reverse=True)[:n]

    return {
        "value": sorted_tcs,
        "detail": f"Top {n} of {len(tc_avgs)} unique test cases",
    }


def calc_completeness_velocity(data: List[Dict]) -> Dict[str, Any]:
    """
    Completeness Velocity: how fast the completeness score climbs per turn.
    Requires per-turn score data (stored in 'per_turn_scores' field).
    """
    velocities = []
    for d in data:
        scores = d.get("per_turn_scores", [])
        if len(scores) >= 2:
            # Velocity = average score increase per turn
            deltas = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
            avg_delta = sum(deltas) / len(deltas)
            velocities.append(avg_delta)

    if not velocities:
        return {
            "value": "N/A",
            "detail": "No per-turn score data available. Enable per-turn tracking to use this KPI.",
        }

    avg_velocity = sum(velocities) / len(velocities)
    return {
        "value": round(avg_velocity, 1),
        "unit": "pts/turn",
        "detail": f"Avg score increase per turn across {len(velocities)} conversations",
    }


def calc_premature_ending_rate(data: List[Dict]) -> Dict[str, Any]:
    """
    Premature Ending Rate: % of conversations where the final score < 70
    but the conversation ended (low completeness at close).
    Uses overall_score as proxy when per-turn data isn't available.
    """
    ended_low = 0
    total = 0
    for d in data:
        score = d.get("overall_score")
        result = d.get("result")
        if score is not None and result is not None:
            total += 1
            # A conversation that scored poorly overall but still "completed"
            if score < 50 and result == "FAIL":
                ended_low += 1

    if total == 0:
        return {"value": None, "detail": "No data"}

    return {
        "value": round(ended_low / total * 100, 1),
        "unit": "%",
        "detail": f"{ended_low}/{total} conversations failed with score < 50",
    }


# =====================================
# COMPARISON
# =====================================

def compare_runs(data_a: List[Dict], data_b: List[Dict], label_a: str, label_b: str) -> Dict[str, Any]:
    """Compare two test runs and compute deltas."""
    def avg_score(data):
        scores = [d["overall_score"] for d in data if d["overall_score"] is not None]
        return round(sum(scores) / len(scores), 1) if scores else None

    def pass_rate(data):
        results = [d["result"] for d in data if d["result"]]
        if not results:
            return None
        return round(sum(1 for r in results if r == "PASS") / len(results) * 100, 1)

    def avg_turns(data):
        turns = [d["num_turns"] for d in data if d["num_turns"] is not None]
        return round(sum(turns) / len(turns), 1) if turns else None

    def kpi_avgs(data):
        kpi_scores = defaultdict(list)
        for d in data:
            for kpi_name, score in d.get("kpis", {}).items():
                if score is not None:
                    kpi_scores[kpi_name].append(score)
        return {k: round(sum(v) / len(v), 1) for k, v in kpi_scores.items() if v}

    score_a, score_b = avg_score(data_a), avg_score(data_b)
    pass_a, pass_b = pass_rate(data_a), pass_rate(data_b)
    turns_a, turns_b = avg_turns(data_a), avg_turns(data_b)
    kpis_a, kpis_b = kpi_avgs(data_a), kpi_avgs(data_b)

    comparison = {
        "runs": {label_a: len(data_a), label_b: len(data_b)},
        "avg_score": {
            label_a: score_a, label_b: score_b,
            "delta": round(score_b - score_a, 1) if score_a and score_b else None,
        },
        "pass_rate": {
            label_a: pass_a, label_b: pass_b,
            "delta": round(pass_b - pass_a, 1) if pass_a is not None and pass_b is not None else None,
        },
        "avg_turns": {
            label_a: turns_a, label_b: turns_b,
            "delta": round(turns_b - turns_a, 1) if turns_a and turns_b else None,
        },
        "kpi_deltas": {},
    }

    all_kpis = set(list(kpis_a.keys()) + list(kpis_b.keys()))
    for kpi in sorted(all_kpis):
        a_val = kpis_a.get(kpi)
        b_val = kpis_b.get(kpi)
        comparison["kpi_deltas"][KPI_DISPLAY.get(kpi, kpi)] = {
            label_a: a_val,
            label_b: b_val,
            "delta": round(b_val - a_val, 1) if a_val is not None and b_val is not None else None,
        }

    return comparison


# =====================================
# REPORT FORMATTING
# =====================================

def format_report(kpis: Dict[str, Any], run_label: str) -> str:
    """Format KPI results into a readable text report."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  KPI REPORT — {run_label}")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # Section A: Quality
    lines.append("\n📊 A) QUALITY KPIs")
    lines.append("-" * 50)

    pr = kpis.get("pass_rate", {})
    lines.append(f"  Pass Rate:               {pr.get('value', 'N/A')}{pr.get('unit', '')}")
    lines.append(f"    {pr.get('detail', '')}")

    avg = kpis.get("avg_score", {})
    lines.append(f"  Avg Evaluation Score:    {avg.get('value', 'N/A')}{avg.get('unit', '')}")
    lines.append(f"    {avg.get('detail', '')}")

    dist = kpis.get("score_distribution", {})
    dist_val = dist.get("value")
    if isinstance(dist_val, dict):
        lines.append("  Score Distribution:")
        for band, count in dist_val.items():
            lines.append(f"    {band:25s} {count}")

    weak = kpis.get("weakest_kpi", {})
    lines.append(f"  Weakest KPI:             {weak.get('value', 'N/A')}")
    lines.append(f"    {weak.get('detail', '')}")
    all_avgs = weak.get("all_averages")
    if all_avgs:
        lines.append("  All KPI Averages:")
        for kpi_name, score in all_avgs.items():
            bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
            lines.append(f"    {kpi_name:40s} {score:5.1f}  {bar}")

    safety = kpis.get("safety_compliance", {})
    lines.append(f"  Safety Compliance:       {safety.get('value', 'N/A')}{safety.get('unit', '')}")
    lines.append(f"    {safety.get('detail', '')}")

    wf = kpis.get("workflow_compliance", {})
    lines.append(f"  Workflow Compliance:     {wf.get('value', 'N/A')}{wf.get('unit', '')}")
    lines.append(f"    {wf.get('detail', '')}")

    # Section B: Efficiency
    lines.append("\n⚡ B) EFFICIENCY KPIs")
    lines.append("-" * 50)

    turns = kpis.get("avg_turns", {})
    lines.append(f"  Avg Turns:               {turns.get('value', 'N/A')} {turns.get('unit', '')}")
    lines.append(f"    {turns.get('detail', '')}")

    dur = kpis.get("avg_duration", {})
    lines.append(f"  Avg Duration:            {dur.get('value', 'N/A')} {dur.get('unit', '')}")
    lines.append(f"    {dur.get('detail', '')}")

    over = kpis.get("overstaying_rate", {})
    lines.append(f"  Overstaying Rate:        {over.get('value', 'N/A')}{over.get('unit', '')}")
    lines.append(f"    {over.get('detail', '')}")

    vel = kpis.get("completeness_velocity", {})
    lines.append(f"  Completeness Velocity:   {vel.get('value', 'N/A')} {vel.get('unit', '')}")
    lines.append(f"    {vel.get('detail', '')}")

    prem = kpis.get("premature_ending_rate", {})
    lines.append(f"  Premature Ending Rate:   {prem.get('value', 'N/A')}{prem.get('unit', '')}")
    lines.append(f"    {prem.get('detail', '')}")

    # Section C: Consistency
    lines.append("\n🔄 C) CONSISTENCY KPIs")
    lines.append("-" * 50)

    fair = kpis.get("persona_fairness", {})
    lines.append(f"  Persona Fairness Gap:    {fair.get('value', 'N/A')} {fair.get('unit', '')}")
    lines.append(f"    {fair.get('detail', '')}")

    # Section D: Topic-Level
    lines.append("\n🎯 D) TOPIC-LEVEL KPIs")
    lines.append("-" * 50)

    worst = kpis.get("worst_test_cases", {})
    worst_val = worst.get("value")
    if isinstance(worst_val, list):
        lines.append(f"  Worst Test Cases ({worst.get('detail', '')}):")
        for name, score in worst_val:
            lines.append(f"    ❌ {score:5.1f}  {name[:60]}")

    best = kpis.get("best_test_cases", {})
    best_val = best.get("value")
    if isinstance(best_val, list):
        lines.append(f"  Best Test Cases ({best.get('detail', '')}):")
        for name, score in best_val:
            lines.append(f"    ✅ {score:5.1f}  {name[:60]}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def format_comparison_report(comparison: Dict[str, Any]) -> str:
    """Format a comparison between two runs."""
    lines = []
    lines.append("=" * 70)
    lines.append("  RUN COMPARISON (Prompt Version Delta)")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    runs = comparison.get("runs", {})
    for label, count in runs.items():
        lines.append(f"  {label}: {count} executions")

    def delta_arrow(d):
        if d is None:
            return "  ?"
        if d > 0:
            return f" ▲ +{d}"
        elif d < 0:
            return f" ▼ {d}"
        return "  ─ 0"

    lines.append("\n📊 Score Comparison")
    lines.append("-" * 50)

    avg = comparison.get("avg_score", {})
    for k, v in avg.items():
        if k != "delta":
            lines.append(f"  {k:20s} {v}")
    lines.append(f"  {'Delta':20s}{delta_arrow(avg.get('delta'))}")

    pr = comparison.get("pass_rate", {})
    lines.append(f"\n  Pass Rate:")
    for k, v in pr.items():
        if k != "delta":
            lines.append(f"  {k:20s} {v}%")
    lines.append(f"  {'Delta':20s}{delta_arrow(pr.get('delta'))}")

    turns = comparison.get("avg_turns", {})
    lines.append(f"\n  Avg Turns:")
    for k, v in turns.items():
        if k != "delta":
            lines.append(f"  {k:20s} {v}")
    lines.append(f"  {'Delta':20s}{delta_arrow(turns.get('delta'))}")

    kpi_deltas = comparison.get("kpi_deltas", {})
    if kpi_deltas:
        lines.append("\n📈 KPI Comparison")
        lines.append("-" * 50)
        lines.append(f"  {'KPI':<40s} {'Delta':>8s}")
        for kpi_name, vals in kpi_deltas.items():
            d = vals.get("delta")
            arrow = delta_arrow(d)
            lines.append(f"  {kpi_name:<40s} {arrow}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# =====================================
# NOTION WRITE-BACK
# =====================================

def write_kpis_to_test_run(test_runs_db_id: str, run_prefix: str, kpis: Dict[str, Any]):
    """Write computed KPI summary back to the Test Runs DB."""
    logger.info(f"Writing KPI summary to Test Run '{run_prefix}'")

    # Find the test run page
    db_uuid = format_notion_id(test_runs_db_id)
    try:
        resp = httpx.post(
            f"{NOTION_API_BASE}/databases/{db_uuid}/query",
            headers=HEADERS,
            json={"filter": {"property": "Test Run Number", "title": {"equals": run_prefix}}},
            timeout=30.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            logger.warning(f"Test Run '{run_prefix}' not found")
            print(f"⚠️ Test Run '{run_prefix}' not found in database")
            return

        test_run_page = results[0]
        test_run_id = test_run_page["id"]

        # Build properties update
        properties: Dict[str, Any] = {}

        # Update Score with the calculated average
        avg = kpis.get("avg_score", {}).get("value")
        if avg is not None:
            properties["Score"] = {"number": float(avg)}

        # Update Total Duration
        dur = kpis.get("avg_duration", {})
        dur_detail = dur.get("detail", "")
        # Extract total minutes from detail
        total_match = re.search(r"Total: ([\d.]+) min", dur_detail)
        if total_match:
            properties["Total Duration"] = {"number": float(total_match.group(1))}

        # Build a KPI summary for Notes
        pr = kpis.get("pass_rate", {})
        turns = kpis.get("avg_turns", {})
        weak = kpis.get("weakest_kpi", {})

        summary_parts = []
        if pr.get("value") is not None:
            summary_parts.append(f"Pass Rate: {pr['value']}%")
        if avg is not None:
            summary_parts.append(f"Avg Score: {avg}/100")
        if turns.get("value") is not None:
            summary_parts.append(f"Avg Turns: {turns['value']}")
        if weak.get("value") and weak["value"] != "N/A":
            summary_parts.append(f"Weakest: {weak['value']}")

        over = kpis.get("overstaying_rate", {})
        if over.get("value") is not None:
            summary_parts.append(f"Overstaying: {over['value']}%")

        if summary_parts:
            summary_text = " | ".join(summary_parts)
            properties["Notes"] = {
                "rich_text": [{"text": {"content": summary_text[:2000]}}]
            }

        if properties:
            resp = httpx.patch(
                f"{NOTION_API_BASE}/pages/{test_run_id}",
                headers=HEADERS,
                json={"properties": properties},
                timeout=30.0,
            )
            resp.raise_for_status()
            logger.info(f"✅ Updated Test Run '{run_prefix}' with KPI summary")
            print(f"✅ Updated Test Run '{run_prefix}' with KPI summary")
        else:
            print("⚠️ No KPI data to write back")

    except Exception as e:
        log_exception(logger, e, "write_kpis_to_test_run")
        print(f"❌ Error writing KPIs to Test Run: {e}")


# =====================================
# MAIN
# =====================================

def main():
    parser = argparse.ArgumentParser(description="Calculate KPIs for chatbot test runs")
    parser.add_argument("--run", type=str, help="Test run prefix (e.g., TR1). Omit for all runs.")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                        help="Compare two test runs (e.g., --compare TR1 TR2)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted report")
    parser.add_argument("--write", action="store_true", help="Write KPI summary back to Notion Test Runs DB")
    parser.add_argument("--no-details", action="store_true",
                        help="Skip fetching detailed KPIs from evaluation pages (faster)")

    args = parser.parse_args()

    # Discover DB IDs
    test_exec_db_id = find_database_in_page(TEST_CASE_EXECUTIONS_PAGE_ID)
    test_runs_db_id = find_database_in_page(TEST_RUNS_PAGE_ID) if TEST_RUNS_PAGE_ID else None

    if not test_exec_db_id:
        print("❌ Could not resolve Test Case Executions DB ID")
        sys.exit(1)

    fetch_details = not args.no_details

    # --- Compare mode ---
    if args.compare:
        run_a, run_b = args.compare
        print(f"📊 Comparing {run_a} vs {run_b}...")

        exec_a = get_evaluated_executions(test_exec_db_id, run_a)
        exec_b = get_evaluated_executions(test_exec_db_id, run_b)

        if not exec_a or not exec_b:
            print(f"❌ Need data for both runs. {run_a}: {len(exec_a)}, {run_b}: {len(exec_b)}")
            sys.exit(1)

        print(f"  {run_a}: {len(exec_a)} executions, {run_b}: {len(exec_b)} executions")
        print("  Fetching evaluation details...")

        data_a = collect_execution_data(exec_a, fetch_kpi_details=fetch_details)
        data_b = collect_execution_data(exec_b, fetch_kpi_details=fetch_details)

        comparison = compare_runs(data_a, data_b, run_a, run_b)

        if args.json:
            print(json.dumps(comparison, indent=2, ensure_ascii=False))
        else:
            print(format_comparison_report(comparison))
        return

    # --- Single run / all runs mode ---
    run_label = args.run or "All Runs"
    print(f"📊 Calculating KPIs for: {run_label}")

    executions = get_evaluated_executions(test_exec_db_id, args.run)
    if not executions:
        print("❌ No evaluated executions found.")
        sys.exit(1)

    print(f"  Found {len(executions)} evaluated executions")
    print("  Collecting data" + (" (with detailed KPIs)..." if fetch_details else "..."))

    data = collect_execution_data(executions, fetch_kpi_details=fetch_details)

    # Calculate all KPIs
    kpis = {
        # Quality
        "pass_rate": calc_pass_rate(data),
        "avg_score": calc_avg_score(data),
        "score_distribution": calc_score_distribution(data),
        "weakest_kpi": calc_weakest_kpi(data),
        "safety_compliance": calc_safety_compliance(data),
        "workflow_compliance": calc_workflow_compliance(data),
        # Efficiency
        "avg_turns": calc_avg_turns(data),
        "avg_duration": calc_avg_duration(data),
        "overstaying_rate": calc_overstaying_rate(data),
        "completeness_velocity": calc_completeness_velocity(data),
        "premature_ending_rate": calc_premature_ending_rate(data),
        # Consistency
        "persona_fairness": calc_persona_fairness(data),
        # Topic-level
        "worst_test_cases": calc_worst_test_cases(data),
        "best_test_cases": calc_best_test_cases(data),
    }

    if args.json:
        print(json.dumps(kpis, indent=2, ensure_ascii=False, default=str))
    else:
        print(format_report(kpis, run_label))

    # Write back to Notion
    if args.write and args.run and test_runs_db_id:
        write_kpis_to_test_run(test_runs_db_id, args.run, kpis)
    elif args.write and not args.run:
        print("⚠️ --write requires --run to specify which test run to update")


if __name__ == "__main__":
    main()
