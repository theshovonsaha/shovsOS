import asyncio
import sys
import os
import json
import pytest

# Add project root to path
sys.path.append(os.getcwd())

from plugins.tools import _pdf_processor

@pytest.mark.asyncio
async def test_pdf_tool():
    print("Testing PDF Processor Tool...")
    
    # 1. Create
    print("\n1. Action: create")
    res_create = await _pdf_processor(
        action="create",
        output_path="test_v10.pdf",
        content="Antigravity V10 Engine\nPDF Processing System\nVerified: 2026-02-26"
    )
    print(f"Create Result: {res_create}")
    
    # 2. Read
    print("\n2. Action: read")
    res_read = await _pdf_processor(
        action="read",
        path="test_v10.pdf"
    )
    print(f"Read Result: {res_read}")
    
    # 3. Split
    print("\n3. Action: split")
    res_split = await _pdf_processor(
        action="split",
        path="test_v10.pdf"
    )
    print(f"Split Result: {res_split}")
    
    # 4. Merge
    print("\n4. Action: merge")
    # Using the same file twice for a simple merge test
    res_merge = await _pdf_processor(
        action="merge",
        paths=["test_v10.pdf", "test_v10.pdf"],
        output_path="merged_test.pdf"
    )
    print(f"Merge Result: {res_merge}")

if __name__ == "__main__":
    asyncio.run(test_pdf_tool())
