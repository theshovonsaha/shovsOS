#!/usr/bin/env python3
"""
Trace Viewer for Shovs LLM OS
Extracts and prints the exact context packets sent to the LLM for debugging.
"""

import sys
import os
import argparse
from typing import Optional

# Ensure we can import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.trace_store import get_trace_store


def main():
    parser = argparse.ArgumentParser(description="View LLM context packets from the Trace Store.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--latest", action="store_true", help="View traces for the most recent run.")
    group.add_argument("--run-id", type=str, help="View traces for a specific run ID.")
    group.add_argument("--session-id", type=str, help="View traces for a specific session ID.")
    
    parser.add_argument("--limit", type=int, default=50, help="Max events to process (default 50).")
    
    args = parser.parse_args()
    
    store = get_trace_store()
    
    run_id = args.run_id
    session_id = args.session_id
    
    if args.latest:
        print("Fetching latest events to determine most recent run...", file=sys.stderr)
        # Fetch a few recent events to find the latest run_id
        recent = store.list_events(limit=50)
        for e in recent:
            rid = e.get("run_id")
            if rid:
                run_id = rid
                break
        
        if not run_id:
            print("No recent run IDs found in the trace store.")
            sys.exit(1)
            
        print(f"Targeting Latest Run ID: {run_id}\n", file=sys.stderr)

    events = store.list_events(run_id=run_id, session_id=session_id, limit=args.limit)
    if not events:
        print(f"No events found matching criteria.")
        sys.exit(0)
        
    # list_events returns in descending chronological order (newest first). 
    # We reverse it to display chronologically.
    events = list(reversed(events))
    
    print("=" * 80)
    print(f"TRACE VIEWER")
    if run_id:
        print(f"Run ID: {run_id}")
    if session_id:
        print(f"Session ID: {session_id}")
    print("=" * 80)
    print()

    for shallow_event in events:
        # Full payload might be on disk; load it.
        event_id = shallow_event.get("id")
        event = store.get_event(event_id)
        if not event:
            continue
            
        event_type = event.get("event_type")
        data = event.get("data") or {}
        
        if event_type == "phase_context":
            phase = data.get("phase", "UNKNOWN_PHASE").upper()
            content = data.get("content", "")
            
            print(f"\n{'=' * 80}")
            print(f"🔹 PHASE CONTEXT: {phase} (Event: {event_id})")
            print(f"{'=' * 80}\n")
            
            if content:
                print(content)
            else:
                print("[No Content / Empty Packet]")
                
        elif event_type == "assistant_response":
            content = data.get("content", "")
            print(f"\n{'-' * 80}")
            print(f"🔸 LLM RESPONSE (Event: {event_id})")
            print(f"{'-' * 80}\n")
            print(content)
            
        elif event_type == "tool_result":
            tool_name = data.get("tool_name", "unknown_tool")
            success = data.get("success", False)
            status = "SUCCESS" if success else "FAILED"
            
            # Use red if failed, green if success (using simple ANSI codes if supported, else just text)
            print(f"\n>> TOOL RESULT: {tool_name} [{status}] <<")
            
        elif event_type == "hard_failure":
            tool_name = data.get("tool_name", "unknown_tool")
            print(f"\n❌ HARD FAILURE TRIGGERED: {tool_name}")


if __name__ == "__main__":
    main()
