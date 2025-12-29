# run_test_evaluations.py
"""
Continuous Test Evaluation Runner
Monitors for test executions with status 'Test Executed' and evaluates them automatically.
Runs in a loop like prepare_test_runs.py
"""
import os
import time
import json
from typing import List, Dict, Any, Optional, Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Optional OpenAI import
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

from logging_config import setup_logging, log_exception, log_api_call

# =====================================
# LOAD .env
# =====================================

load_dotenv()

# Set up logging
logger = setup_logging("test_evaluation")

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
TEST_CASE_EXECUTIONS_PAGE_ID = os.getenv("TEST_CASE_EXECUTIONS_DB_ID")
EVALUATION_PAGES_PARENT_PAGE_ID = os.getenv("EVALUATION_PAGES_DB_ID")
TEST_RUNS_DB_ID = os.getenv("TEST_RUNS_DB_ID")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

# Property names
EVALUATION_TEXT_PROPERTY = os.getenv("EVALUATION_TEXT_PROPERTY", "Evaluation JSON")
EVALUATION_SCORE_PROPERTY = os.getenv("EVALUATION_SCORE_PROPERTY", "Evaluation score")
EVALUATION_PAGE_LINK_PROPERTY = os.getenv("EVALUATION_PAGE_LINK_PROPERTY", "")

# ✅ NEW: Check interval configuration
CHECK_INTERVAL = int(os.getenv("EVALUATION_CHECK_INTERVAL", "10"))  # seconds

openai_client: Optional[OpenAI] = None
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

NOTION_API_BASE = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

logger.info("✓ Starting Test Case Evaluations Runner (Continuous Mode)")
logger.info(f"Check Interval: {CHECK_INTERVAL}s")
print("✓ Starting Test Case Evaluations Runner (Continuous Mode)")


# =====================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =====================================

class KPI(BaseModel):
    score: int = Field(..., ge=1, le=100, description="Score from 1-100")
    comment: str = Field(..., description="Justification with at least one improvement suggestion")


class KPIs(BaseModel):
    task_completeness: KPI
    user_comfort: KPI
    understanding_and_relevance: KPI
    clarity_and_actionability: KPI
    edge_cases_and_constraints: KPI
    proactiveness_and_guidance: KPI
    tone_and_personalization: KPI
    accuracy_and_policy_compliance: KPI
    efficiency_and_flow: KPI


class Overall(BaseModel):
    overall_score: int = Field(..., ge=1, le=100, description="Overall score from 1-100")
    result: Literal["PASS", "PARTIAL", "FAIL"] = Field(..., description="Must be exactly PASS, PARTIAL, or FAIL")
    comment: str = Field(..., description="One paragraph with strengths and improvements")


class EvaluationOutput(BaseModel):
    summary_of_goal: str = Field(..., description="1-3 sentences describing user goals and how well addressed")
    kpis: KPIs
    overall: Overall


# =====================================
# NOTION HELPERS
# =====================================

def format_notion_id(notion_id: str) -> str:
    clean = (notion_id or "").replace("-", "")
    if len(clean) == 32:
        return f"{clean[0:8]}-{clean[8:12]}-{clean[12:16]}-{clean[16:20]}-{clean[20:32]}"
    return notion_id


def notion_page_url(page_id: str) -> str:
    clean = (page_id or "").replace("-", "")
    return f"https://www.notion.so/{clean}"


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
        data = resp.json()

        for block in data.get("results", []):
            if block.get("type") == "child_database":
                return block.get("id")

        return page_uuid
    except Exception as e:
        logger.warning(f"⚠️ Error discovering DB in page {page_id}: {e}")
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
            if not lst:
                return ""
            full_text = "".join(item.get("plain_text", "") for item in lst)
            return full_text
        elif ptype == "number":
            return prop.get("number")
        elif ptype == "select":
            sel = prop.get("select")
            return sel["name"] if sel else None
        elif ptype == "status":
            st = prop.get("status")
            return st["name"] if st else None
        elif ptype == "relation":
            return prop.get("relation", [])
        elif ptype == "date":
            d = prop.get("date")
            return d["start"] if d else None
        else:
            return None
    except Exception as e:
        logger.warning(f"⚠️ Error extracting property '{property_name}': {e}")
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
    except Exception as e:
        logger.warning(f"⚠️ Error extracting link from '{property_name}': {e}")
        return None


def extract_page_id_from_url(url: str) -> Optional[str]:
    if not url:
        return None

    base = url.split("?", 1)[0].split("#", 1)[0]
    parts = base.rstrip("/").split("/")
    if not parts:
        return None

    last = parts[-1]
    import re
    hex_only = "".join(re.findall(r"[0-9a-fA-F]", last))
    if len(hex_only) < 32:
        return None

    clean = hex_only[-32:]
    return format_notion_id(clean)


def load_text_from_page_blocks(page_id: str) -> str:
    """Load text from a Notion page, handling the new formatted conversation structure."""
    page_uuid = format_notion_id(page_id)
    lines: List[str] = []
    cursor = None

    def extract_rich_text(rich_text_list):
        return "".join(rt.get("plain_text", "") for rt in rich_text_list)

    def process_blocks(blocks):
        for block in blocks:
            btype = block.get("type")
            inner = block.get(btype, {})
            
            if btype in ("paragraph", "heading_1", "heading_2", "heading_3"):
                rich = inner.get("rich_text", [])
                txt = extract_rich_text(rich)
                if txt.strip():
                    lines.append(txt)
            
            elif btype == "callout":
                rich = inner.get("rich_text", [])
                txt = extract_rich_text(rich)
                if txt.strip():
                    lines.append(txt)
            
            elif btype == "toggle":
                rich = inner.get("rich_text", [])
                txt = extract_rich_text(rich)
                if txt.strip():
                    lines.append(txt)
                
                children = inner.get("children", [])
                if children:
                    process_blocks(children)
            
            elif btype in ("bulleted_list_item", "numbered_list_item"):
                rich = inner.get("rich_text", [])
                txt = extract_rich_text(rich)
                if txt.strip():
                    lines.append(txt)

    while True:
        params: Dict[str, Any] = {}
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

        process_blocks(data.get("results", []))

        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")

    full_text = "\n\n".join(lines).strip()
    
    logger.debug(f"   📄 Extracted {len(full_text)} characters from conversation page")
    
    return full_text


def find_prop_key_ci(props: Dict[str, Any], desired_name: str) -> Optional[str]:
    """Case-insensitive property name match."""
    if not desired_name:
        return None
    for k in props.keys():
        if k.lower() == desired_name.lower():
            return k
    return None


# =====================================
# QUERY EXECUTIONS TO EVALUATE
# =====================================

def get_executions_to_evaluate(db_id: str) -> List[Dict[str, Any]]:
    """Query for executions with status 'Test Executed'"""
    logger.info("Querying for executions to evaluate")
    results: List[Dict[str, Any]] = []
    cursor = None

    while True:
        body: Dict[str, Any] = {
            "filter": {
                "property": "Test Execution Status",
                "status": {"equals": "Test Executed"},
            }
        }
        if cursor:
            body["start_cursor"] = cursor

        resp = httpx.post(
            f"{NOTION_API_BASE}/databases/{db_id}/query",
            headers=HEADERS,
            json=body,
            timeout=30.0,
        )

        if resp.status_code >= 400:
            logger.error(f"❌ Error querying executions DB (status {resp.status_code})")
            resp.raise_for_status()

        data = resp.json()
        results.extend(data.get("results", []))

        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")

    logger.info(f"✓ Found {len(results)} executions to evaluate")
    return results


# =====================================
# TEST RUN STATISTICS UPDATE
# =====================================

def find_test_run_by_number(db_id: str, test_run_number: str) -> Optional[Dict[str, Any]]:
    """Find a Test Run page by Test Run Number."""
    if not db_id or not test_run_number:
        return None
    
    db_uuid = format_notion_id(db_id)
    
    try:
        logger.debug(f"   🔍 Searching Test Runs DB for '{test_run_number}'...")
        resp = httpx.post(
            f"{NOTION_API_BASE}/databases/{db_uuid}/query",
            headers=HEADERS,
            json={
                "filter": {
                    "property": "Test Run Number",
                    "title": {
                        "equals": test_run_number
                    }
                }
            },
            timeout=30.0,
        )
        
        if resp.status_code == 400:
            # Try rich_text filter
            resp = httpx.post(
                f"{NOTION_API_BASE}/databases/{db_uuid}/query",
                headers=HEADERS,
                json={
                    "filter": {
                        "property": "Test Run Number",
                        "rich_text": {
                            "equals": test_run_number
                        }
                    }
                },
                timeout=30.0,
            )
        
        if resp.status_code >= 400:
            # Last resort: query all and filter manually
            resp = httpx.post(
                f"{NOTION_API_BASE}/databases/{db_uuid}/query",
                headers=HEADERS,
                json={},
                timeout=30.0,
            )
            
            if resp.status_code >= 400:
                return None
            
            data = resp.json()
            results = data.get("results", [])
            
            for page in results:
                title_prop = page.get("properties", {}).get("Test Run Number", {})
                title_list = title_prop.get("title", [])
                if title_list:
                    page_title = title_list[0].get("plain_text", "")
                    if page_title == test_run_number:
                        return page
            
            return None
        
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        
        if results:
            logger.debug(f"   ✅ Found Test Run '{test_run_number}'")
            return results[0]
        
        return None
        
    except Exception as e:
        logger.warning(f"   ❌ Error searching for Test Run '{test_run_number}': {e}")
        return None


def update_test_run_statistics(
    test_runs_db_id: str,
    test_run_number: str,
    evaluation_score: float,
    passed: bool
):
    """Update Test Run statistics based on evaluation result."""
    if not test_runs_db_id or not test_run_number:
        logger.debug("⚠️ Test Runs DB ID or test run number missing, skipping statistics update")
        return
    
    # Extract just the Test Run prefix (e.g., "TR2" from "TR2.TC16.1")
    test_run_prefix = test_run_number.split(".")[0] if "." in test_run_number else test_run_number
    logger.debug(f"   📊 Updating Test Run '{test_run_prefix}' (from execution '{test_run_number}')")
    
    # Find the Test Run page
    test_run_page = find_test_run_by_number(test_runs_db_id, test_run_prefix)
    if not test_run_page:
        logger.warning(f"⚠️ Test Run '{test_run_prefix}' not found in Test Runs database")
        return
    
    test_run_id = test_run_page.get("id")
    
    # Extract current values
    succeeded = extract_property_value(test_run_page, "Number of test cases succeeded") or 0
    failed = extract_property_value(test_run_page, "Number of test cases failed") or 0
    current_score = extract_property_value(test_run_page, "Score") or 0
    
    # Calculate new values
    if passed:
        new_succeeded = int(succeeded) + 1
        new_failed = int(failed)
    else:
        new_succeeded = int(succeeded)
        new_failed = int(failed) + 1
    
    # Calculate new average score
    completed_count = int(succeeded) + int(failed)
    total_score = current_score * completed_count if completed_count > 0 else 0
    new_average = (total_score + evaluation_score) / (completed_count + 1)
    
    # Update the Test Run page
    properties = {
        "Number of test cases succeeded": {"number": new_succeeded},
        "Number of test cases failed": {"number": new_failed},
        "Score": {"number": round(new_average, 2)}
    }
    
    try:
        resp = httpx.patch(
            f"{NOTION_API_BASE}/pages/{test_run_id}",
            headers=HEADERS,
            json={"properties": properties},
            timeout=30.0,
        )
        resp.raise_for_status()
        logger.info(f"   ✅ Updated Test Run '{test_run_prefix}':")
        logger.info(f"      Succeeded: {succeeded} → {new_succeeded}")
        logger.info(f"      Failed: {failed} → {new_failed}")
        logger.info(f"      Average Score: {current_score:.2f} → {new_average:.2f}")
    except Exception as e:
        logger.error(f"❌ Error updating Test Run statistics: {e}")


# =====================================
# OPENAI EVALUATION
# =====================================

def fill_evaluation_prompt_template(
    template: str,
    test_case_name: str,
    persona: str,
    test_case_details: str,
    evaluation_criteria: str,
    conversation_text: str,
) -> str:
    template = template or ""
    return (
        template
        .replace("{{test_case}}", test_case_name or "")
        .replace("{{persona}}", persona or "")
        .replace("{{test_case_details}}", test_case_details or "")
        .replace("{{evaluation_criteria}}", evaluation_criteria or "")
        .replace("{{conversation_flow}}", conversation_text or "")
    )


def call_evaluator_with_template(
    template_prompt: str,
    test_case_name: str,
    persona: str,
    test_case_details: str,
    evaluation_criteria: str,
    conversation_text: str,
) -> Dict[str, Any]:
    if openai_client is None:
        return {"parsed": None, "raw": "", "error": "OPENAI client unavailable."}
    if not OPENAI_MODEL:
        return {"parsed": None, "raw": "", "error": "OPENAI_MODEL not set in environment."}

    filled_prompt = fill_evaluation_prompt_template(
        template=template_prompt,
        test_case_name=test_case_name,
        persona=persona,
        test_case_details=test_case_details,
        evaluation_criteria=evaluation_criteria,
        conversation_text=conversation_text,
    )

    try:
        resp = openai_client.beta.chat.completions.parse(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict evaluation engine. Follow all instructions carefully and provide detailed, critical analysis."
                },
                {"role": "user", "content": filled_prompt},
            ],
            response_format=EvaluationOutput,
        )

        parsed_model = resp.choices[0].message.parsed
        
        if parsed_model is None:
            refusal = resp.choices[0].message.refusal
            return {
                "parsed": None,
                "raw": "",
                "error": f"Model refused or failed to parse: {refusal}"
            }

        parsed_dict = parsed_model.model_dump()
        raw_json = json.dumps(parsed_dict, indent=2, ensure_ascii=False)

        return {"parsed": parsed_dict, "raw": raw_json, "error": None}

    except Exception as e:
        return {"parsed": None, "raw": "", "error": f"OpenAI error: {e}"}


# =====================================
# NOTION BLOCK BUILDERS
# =====================================

def score_to_bg_color(score: Optional[float]) -> str:
    if score is None:
        return "gray_background"
    if score < 30:
        return "red_background"
    if score < 60:
        return "yellow_background"
    return "green_background"


def rich_text(content: str, color: str = "default", link_url: Optional[str] = None) -> Dict[str, Any]:
    obj: Dict[str, Any] = {
        "type": "text",
        "text": {"content": content},
        "annotations": {
            "bold": False,
            "italic": False,
            "strikethrough": False,
            "underline": False,
            "code": False,
            "color": color,
        },
    }
    if link_url:
        obj["text"]["link"] = {"url": link_url}
    return obj


def heading_block(text: str, level: int = 2) -> Dict[str, Any]:
    t = f"heading_{level}"
    return {"object": "block", "type": t, t: {"rich_text": [rich_text(text)]}}


def paragraph_block(text: str) -> Dict[str, Any]:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [rich_text(text)]}}


def code_block(code: str, language: str = "json") -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "code",
        "code": {
            "rich_text": [rich_text(code)],
            "language": language,
        },
    }


def kpi_table_block(kpis_dict: Dict[str, Any]) -> Dict[str, Any]:
    header_row = {
        "object": "block",
        "type": "table_row",
        "table_row": {
            "cells": [
                [rich_text("KPI")],
                [rich_text("Score")],
                [rich_text("Comment")],
            ]
        },
    }

    rows = [header_row]

    kpi_names = [
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

    for kpi_name in kpi_names:
        kpi_data = kpis_dict.get(kpi_name, {})
        score = kpi_data.get("score")
        comment = kpi_data.get("comment", "")

        try:
            s_val = float(score) if score is not None else None
        except (TypeError, ValueError):
            s_val = None

        bg = score_to_bg_color(s_val)
        score_text = str(score) if score is not None else "-"
        display_name = kpi_name.replace("_", " ").title()

        rows.append(
            {
                "object": "block",
                "type": "table_row",
                "table_row": {
                    "cells": [
                        [rich_text(display_name)],
                        [rich_text(score_text, color=bg)],
                        [rich_text(comment)],
                    ]
                },
            }
        )

    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": 3,
            "has_column_header": True,
            "has_row_header": False,
            "children": rows,
        },
    }


def split_text_for_notion(text: str, max_length: int = 1800) -> List[str]:
    if not text:
        return [""]
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


# =====================================
# CREATE EVALUATION PAGE
# =====================================

def create_evaluation_page_under_parent(
    parent_page_id: str,
    title: str,
    evaluation_json: Dict[str, Any],
) -> Dict[str, Any]:
    parent_uuid = format_notion_id(parent_page_id)

    summary_of_goal = str(evaluation_json.get("summary_of_goal", "")).strip()
    kpis = evaluation_json.get("kpis", {})
    overall = evaluation_json.get("overall", {})

    blocks: List[Dict[str, Any]] = []
    blocks.append(heading_block("Evaluation", level=2))

    # Summary of Goal
    if summary_of_goal:
        blocks.append(heading_block("Summary of Goal", level=3))
        for chunk in split_text_for_notion(summary_of_goal):
            blocks.append(paragraph_block(chunk))

    # Overall Result
    blocks.append(heading_block("Overall Result", level=3))
    overall_score = overall.get("overall_score")
    result = overall.get("result", "")
    overall_comment = overall.get("comment", "")
    
    result_emoji = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌"}.get(result, "")
    blocks.append(paragraph_block(f"{result_emoji} {result} - Score: {overall_score}/100"))
    if overall_comment:
        for chunk in split_text_for_notion(overall_comment):
            blocks.append(paragraph_block(chunk))

    # KPI scores table
    blocks.append(heading_block("KPI Scores", level=3))
    blocks.append(kpi_table_block(kpis))

    # Raw JSON
    blocks.append(heading_block("Raw evaluation JSON", level=3))
    pretty_json = json.dumps(evaluation_json, indent=2, ensure_ascii=False)
    for chunk in split_text_for_notion(pretty_json, max_length=1800):
        blocks.append(code_block(chunk, language="json"))

    payload = {
        "parent": {"page_id": parent_uuid},
        "properties": {"title": [{"type": "text", "text": {"content": title}}]},
        "children": blocks,
    }

    resp = httpx.post(
        f"{NOTION_API_BASE}/pages",
        headers=HEADERS,
        json=payload,
        timeout=60.0,
    )

    if resp.status_code >= 400:
        logger.error(f"❌ Failed to create evaluation page (status {resp.status_code})")
        resp.raise_for_status()

    created = resp.json()
    return {"page_id": created.get("id"), "url": created.get("url")}


# =====================================
# UPDATE EXECUTION ROW
# =====================================

def extract_overall_score_from_parsed(parsed_json: Any) -> Optional[float]:
    if not isinstance(parsed_json, dict):
        return None
    
    overall = parsed_json.get("overall", {})
    if isinstance(overall, dict):
        score = overall.get("overall_score")
        if score is not None:
            try:
                return float(score)
            except (TypeError, ValueError):
                pass
    
    return None


def mark_evaluation_started(page_id: str):
    resp = httpx.patch(
        f"{NOTION_API_BASE}/pages/{page_id}",
        headers=HEADERS,
        json={"properties": {"Test Execution Status": {"status": {"name": "Evaluation started"}}}},
        timeout=30.0,
    )
    if resp.status_code >= 400:
        logger.warning("⚠️ Could not set 'Evaluation started'")
        resp.raise_for_status()
    logger.debug("   ✓ Marked as 'Evaluation started'")


def update_execution_row_after_evaluation(
    execution_page_id: str,
    execution_properties: Dict[str, Any],
    evaluation_json: Optional[Dict[str, Any]],
    raw_json: str,
    error: Optional[str],
    evaluation_page_id: Optional[str],
):
    props: Dict[str, Any] = {
        "Test Execution Status": {"status": {"name": "Evaluation completed"}}
    }

    eval_url = notion_page_url(evaluation_page_id) if evaluation_page_id else None

    txt_key = find_prop_key_ci(execution_properties, EVALUATION_TEXT_PROPERTY)

    if txt_key:
        txt_type = execution_properties.get(txt_key, {}).get("type")
        if txt_type == "rich_text":
            if eval_url:
                props[txt_key] = {"rich_text": [rich_text("Open evaluation page", link_url=eval_url)]}

    # Score
    score = extract_overall_score_from_parsed(evaluation_json) if evaluation_json else None
    score_key = find_prop_key_ci(execution_properties, EVALUATION_SCORE_PROPERTY)
    if score is not None and score_key:
        stype = execution_properties.get(score_key, {}).get("type")
        if stype == "number":
            props[score_key] = {"number": float(score)}

    # Optional link property
    if evaluation_page_id and EVALUATION_PAGE_LINK_PROPERTY:
        link_key = find_prop_key_ci(execution_properties, EVALUATION_PAGE_LINK_PROPERTY)
        if link_key:
            ltype = execution_properties.get(link_key, {}).get("type")
            if ltype == "rich_text":
                props[link_key] = {
                    "rich_text": [rich_text("Open evaluation page", link_url=eval_url)]
                }

    resp = httpx.patch(
        f"{NOTION_API_BASE}/pages/{execution_page_id}",
        headers=HEADERS,
        json={"properties": props},
        timeout=30.0,
    )
    if resp.status_code >= 400:
        logger.error(f"❌ Failed to update execution row (status {resp.status_code})")
        resp.raise_for_status()

    logger.debug(f"   ✓ Updated execution {execution_page_id} → Evaluation completed")


# =====================================
# SINGLE EVALUATION PROCESSOR
# =====================================

def process_single_evaluation(execution: Dict[str, Any], test_exec_db_id: str, test_runs_db_id: Optional[str]) -> bool:
    """
    Process a single test evaluation.
    Returns True if successful, False otherwise.
    """
    page_id = execution.get("id")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"▶ Evaluating execution {page_id}")
    logger.info(f"{'='*60}")
    print(f"\n{'='*60}")
    print(f"▶ Evaluating execution {page_id}")
    print(f"{'='*60}")

    test_case_name = extract_property_value(execution, "Test case name") or ""
    persona = extract_property_value(execution, "Persona") or ""
    test_case_details = extract_property_value(execution, "Test Case Details") or ""
    evaluation_criteria = extract_property_value(execution, "Evaluation Criteria") or ""
    evaluation_template = extract_property_value(execution, "Test Case Evaluation Prompt") or ""
    test_run_number = extract_property_value(execution, "Test run number") or ""

    logger.info(f"   Test case: {test_case_name}")
    logger.info(f"   Persona: {persona}")
    logger.info(f"   Test run: {test_run_number}")
    print(f"   Test case: {test_case_name}")
    print(f"   Persona: {persona}")
    print(f"   Test run: {test_run_number}")

    conv_url = extract_rich_text_link(execution, "Conversation flow")
    if not conv_url:
        logger.warning("   ⚠️ No Conversation flow link found; skipping.")
        print("   ⚠️ No Conversation flow link found; skipping.")
        return False

    conv_page_id = extract_page_id_from_url(conv_url)
    if not conv_page_id:
        logger.warning(f"   ⚠️ Could not parse page ID from conversation URL: {conv_url}")
        print(f"   ⚠️ Could not parse page ID from conversation URL: {conv_url}")
        return False

    conversation_text = load_text_from_page_blocks(conv_page_id)
    logger.debug(f"   🧵 Conversation text length: {len(conversation_text)} chars")
    print(f"   🧵 Conversation text length: {len(conversation_text)} chars")

    mark_evaluation_started(page_id)

    start_ts = time.monotonic()
    eval_result = call_evaluator_with_template(
        template_prompt=evaluation_template,
        test_case_name=test_case_name,
        persona=persona,
        test_case_details=test_case_details,
        evaluation_criteria=evaluation_criteria,
        conversation_text=conversation_text,
    )
    duration = time.monotonic() - start_ts
    logger.info(f"   ⏱ OpenAI evaluation took {duration:.1f}s")
    print(f"   ⏱ OpenAI evaluation took {duration:.1f}s")

    parsed = eval_result.get("parsed")
    raw = eval_result.get("raw", "")
    error = eval_result.get("error")

    evaluation_page_id = None
    if parsed and isinstance(parsed, dict):
        created = create_evaluation_page_under_parent(
            parent_page_id=EVALUATION_PAGES_PARENT_PAGE_ID,
            title=f"{test_case_name} — Evaluation",
            evaluation_json=parsed,
        )
        evaluation_page_id = created.get("page_id")
        logger.info(f"   ✅ Created evaluation page: {created.get('url')}")
        print(f"   ✅ Created evaluation page: {created.get('url')}")
    else:
        logger.warning(f"   ⚠️ No parsed evaluation JSON. Error: {error}")
        print(f"   ⚠️ No parsed evaluation JSON. Error: {error}")

    update_execution_row_after_evaluation(
        execution_page_id=page_id,
        execution_properties=execution.get("properties", {}) or {},
        evaluation_json=parsed if isinstance(parsed, dict) else None,
        raw_json=raw,
        error=error,
        evaluation_page_id=evaluation_page_id,
    )

    # Update Test Run statistics
    if parsed and isinstance(parsed, dict) and test_runs_db_id:
        overall_score = extract_overall_score_from_parsed(parsed)
        if overall_score is not None:
            passed = overall_score > 80
            update_test_run_statistics(
                test_runs_db_id=test_runs_db_id,
                test_run_number=test_run_number,
                evaluation_score=overall_score,
                passed=passed
            )

    logger.info(f"✓ Evaluation {page_id} completed successfully")
    return True


# =====================================
# CONTINUOUS MONITORING LOOP
# =====================================

def verify_env() -> bool:
    """Verify required environment variables"""
    missing = []
    if not NOTION_API_KEY:
        missing.append("NOTION_API_KEY")
    if not TEST_CASE_EXECUTIONS_PAGE_ID:
        missing.append("TEST_CASE_EXECUTIONS_DB_ID")
    if not EVALUATION_PAGES_PARENT_PAGE_ID:
        missing.append("EVALUATION_PAGES_DB_ID (Evaluation-pages PAGE id)")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not OPENAI_MODEL:
        missing.append("OPENAI_MODEL")

    if missing:
        logger.error("❌ Missing required environment variables:")
        print("❌ Missing required environment variables:")
        for m in missing:
            logger.error(f"   • {m}")
            print(f"   • {m}")
        return False

    if not OPENAI_AVAILABLE:
        logger.error("❌ 'openai' package not installed. Run: pip install openai")
        print("❌ 'openai' package not installed. Run: pip install openai")
        return False

    if not TEST_RUNS_DB_ID:
        logger.warning("⚠️ TEST_RUNS_DB_ID not set - Test Run statistics will not be updated")
        print("⚠️ TEST_RUNS_DB_ID not set - Test Run statistics will not be updated")

    return True


def run_evaluation_loop(check_interval: int = 10):
    """
    Continuously monitor for completed test executions and evaluate them.
    Similar to prepare_test_runs.py
    """
    logger.info("=" * 80)
    logger.info("🚀 Test Evaluation Service Started (Continuous Mode)")
    logger.info("=" * 80)
    print("🚀 Test Evaluation Service Started (Continuous Mode)")
    
    if not verify_env():
        return

    test_exec_db_id = find_database_in_page(TEST_CASE_EXECUTIONS_PAGE_ID)
    logger.info(f"   • Test Case Executions DB: {test_exec_db_id}")
    print(f"   • Test Case Executions DB: {test_exec_db_id}")
    
    test_runs_db_id = None
    if TEST_RUNS_DB_ID:
        test_runs_db_id = find_database_in_page(TEST_RUNS_DB_ID)
        logger.info(f"   • Test Runs DB: {test_runs_db_id}")
        print(f"   • Test Runs DB: {test_runs_db_id}")
    else:
        logger.info(f"   ⚠️ TEST_RUNS_DB_ID not configured - statistics updates disabled")
        print(f"   ⚠️ TEST_RUNS_DB_ID not configured - statistics updates disabled")

    logger.info(f"⏰ Checking every {check_interval} seconds...")
    print(f"⏰ Checking every {check_interval} seconds...\n")
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            logger.debug(f"Check iteration #{iteration}")
            
            executions = get_executions_to_evaluate(test_exec_db_id)
            
            if not executions:
                # Only log to file, print to console with \r to overwrite
                logger.debug("💤 No executions to evaluate")
                print("💤 No executions to evaluate...", end="\r")
            else:
                logger.info(f"🔔 Found {len(executions)} execution(s) to evaluate")
                print(f"\n🔔 Found {len(executions)} execution(s) to evaluate")
                
                for idx, execution in enumerate(executions, start=1):
                    logger.info(f"\n[{idx}/{len(executions)}] Evaluating execution...")
                    print(f"\n[{idx}/{len(executions)}] Evaluating execution...")
                    
                    try:
                        success = process_single_evaluation(execution, test_exec_db_id, test_runs_db_id)
                        if success:
                            logger.info(f"✅ Evaluation {idx}/{len(executions)} completed")
                            print(f"✅ Evaluation {idx}/{len(executions)} completed")
                        else:
                            logger.warning(f"⚠️ Evaluation {idx}/{len(executions)} had issues")
                            print(f"⚠️ Evaluation {idx}/{len(executions)} had issues")
                    except Exception as e:
                        log_exception(logger, e, f"process_evaluation_{idx}")
                        logger.error(f"❌ Error evaluating execution {idx}/{len(executions)}: {e}")
                        print(f"❌ Error evaluating execution {idx}/{len(executions)}: {e}")
                        # Continue with next evaluation even if one fails
                        continue
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            logger.warning("\n👋 Shutting down...")
            print("\n\n👋 Shutting down...")
            break
        except Exception as e:
            log_exception(logger, e, "evaluation loop")
            logger.error(f"\n❌ Error in evaluation loop: {e}")
            print(f"\n❌ Error in evaluation loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(check_interval)


# =====================================
# MAIN
# =====================================

def main():
    logger.info("=" * 80)
    logger.info("MAIN EVALUATION STARTED (Continuous Mode)")
    logger.info("=" * 80)
    
    run_evaluation_loop(check_interval=CHECK_INTERVAL)


if __name__ == "__main__":
    logger.info("Script started from command line")
    
    try:
        main()
        logger.info("✅ Script completed successfully")
    except KeyboardInterrupt:
        logger.warning("⚠️ Script interrupted by user (Ctrl+C)")
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        log_exception(logger, e, "main script execution")
        logger.error(f"❌ Script failed: {e}")
        raise
    finally:
        logger.info("Script execution finished")