#!/usr/bin/env python3
"""
check_notion_prompts.py
Quick diagnostic script to check prompt lengths in Notion
"""
import os
from dotenv import load_dotenv
import httpx

load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
TEST_CASE_EXECUTIONS_PAGE_ID = os.getenv("TEST_CASE_EXECUTIONS_DB_ID")

NOTION_API_BASE = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

def format_notion_id(notion_id: str) -> str:
    clean_id = notion_id.replace("-", "")
    if len(clean_id) == 32:
        return f"{clean_id[0:8]}-{clean_id[8:12]}-{clean_id[12:16]}-{clean_id[16:20]}-{clean_id[20:32]}"
    return notion_id

def find_database_in_page(page_id: str):
    page_id = format_notion_id(page_id)
    try:
        response = httpx.get(
            f"{NOTION_API_BASE}/blocks/{page_id}/children",
            headers=HEADERS,
            timeout=30.0
        )
        response.raise_for_status()
        blocks = response.json()
        
        for block in blocks.get("results", []):
            if block.get("type") in ["child_database", "table"]:
                return block["id"]
        return page_id
    except Exception:
        return page_id

def extract_rich_text_all_chunks(prop):
    """Extract and concatenate all rich_text chunks"""
    lst = prop.get("rich_text", [])
    if not lst:
        return ""
    full_text = "".join(item.get("plain_text", "") for item in lst)
    return full_text

def check_prompts():
    print("=" * 80)
    print("CHECKING NOTION PROMPT LENGTHS")
    print("=" * 80)
    
    # Find database
    db_id = find_database_in_page(TEST_CASE_EXECUTIONS_PAGE_ID)
    print(f"Database ID: {db_id}\n")
    
    # Query first few executions
    response = httpx.post(
        f"{NOTION_API_BASE}/databases/{db_id}/query",
        headers=HEADERS,
        json={"page_size": 5},
        timeout=30.0
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    
    print(f"Checking {len(results)} test case executions:\n")
    
    for i, page in enumerate(results, 1):
        props = page["properties"]
        
        # Get test case name
        test_case_name = ""
        if "Test case name" in props:
            test_case_name = extract_rich_text_all_chunks(props["Test case name"])
        
        # Get test run number
        test_run_number = ""
        if "Test run number" in props:
            title_list = props["Test run number"].get("title", [])
            test_run_number = title_list[0]["plain_text"] if title_list else ""
        
        print(f"{i}. {test_run_number} - {test_case_name}")
        
        # Check Test Case Prompt
        if "Test Case Prompt" in props:
            prompt_prop = props["Test Case Prompt"]
            chunks = prompt_prop.get("rich_text", [])
            full_prompt = extract_rich_text_all_chunks(prompt_prop)
            
            print(f"   📝 Test Case Prompt:")
            print(f"      Chunks: {len(chunks)}")
            print(f"      Total length: {len(full_prompt)} chars")
            
            if len(chunks) > 1:
                for j, chunk in enumerate(chunks, 1):
                    chunk_text = chunk.get("plain_text", "")
                    print(f"      Chunk {j}: {len(chunk_text)} chars")
        else:
            print(f"   ⚠️  No 'Test Case Prompt' property found")
        
        # Check Test Case Evaluation Prompt
        if "Test Case Evaluation Prompt" in props:
            eval_prop = props["Test Case Evaluation Prompt"]
            chunks = eval_prop.get("rich_text", [])
            full_eval = extract_rich_text_all_chunks(eval_prop)
            
            print(f"   📊 Test Case Evaluation Prompt:")
            print(f"      Chunks: {len(chunks)}")
            print(f"      Total length: {len(full_eval)} chars")
        else:
            print(f"   ⚠️  No 'Test Case Evaluation Prompt' property found")
        
        # Check other properties
        if "Test Case Details" in props:
            details = extract_rich_text_all_chunks(props["Test Case Details"])
            print(f"   📋 Test Case Details: {len(details)} chars")
        
        if "Persona" in props:
            persona = extract_rich_text_all_chunks(props["Persona"])
            print(f"   👤 Persona: {len(persona)} chars")
        
        print()
    
    print("=" * 80)
    print("CHECK COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    try:
        check_prompts()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()