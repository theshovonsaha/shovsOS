"""Quick classification check — run with: python scripts/check_routes.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.core import _classify_route

tests = [
    # Should now route to multi_step (was previously direct_fact)
    ("search top performing stocks today", "multi_step"),
    ("find the latest news on NVDA", "multi_step"),
    ("look up current weather in NYC", "multi_step"),
    ("fetch me the latest earnings data", "multi_step"),
    ("gather intel on AI companies", "multi_step"),
    ("investigate what happened today", "multi_step"),

    # Should still be direct_fact (no search/fetch/find keyword)
    ("what is the price of gold today", "direct_fact"),
    ("who is the CEO of Apple", "direct_fact"),
    ("current temperature in Tokyo", "direct_fact"),

    # Should still be multi_step
    ("research and summarize AAPL", "multi_step"),
    ("compare NVDA vs AMD", "multi_step"),

    # Still trivial
    ("hey bro", "trivial_chat"),
    ("hi there", "trivial_chat"),
]

ok = 0
for query, expected in tests:
    actual = _classify_route(query, session_has_history=False, current_fact_count=0, active_task_count=0)
    mark = "OK" if actual == expected else "FAIL"
    if actual != expected:
        print(f"  {mark}  '{query}' -> {actual} (expected {expected})")
    else:
        ok += 1
        print(f"  {mark}  '{query}' -> {actual}")

print(f"\n{ok}/{len(tests)} passed")
