"""
Tool Circuit Breaker — Phase 4: Production Readiness
--------------------------------------------------
Prevents infinite loops and wasted turns by tracking tool failures.
If a tool fails N times consecutively, it 'opens' the circuit,
preventing further execution and forcing a pivot.
"""

from typing import Dict, List
from config.logger import log

class CircuitBreaker:
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        # session_id -> {tool_name -> failure_count}
        self.failures: Dict[str, Dict[str, int]] = {}

    def record_failure(self, session_id: str, tool_name: str) -> bool:
        """
        Record a failure for a tool. 
        Returns True if the circuit is now OPEN (too many failures).
        """
        if session_id not in self.failures:
            self.failures[session_id] = {}
        
        count = self.failures[session_id].get(tool_name, 0) + 1
        self.failures[session_id][tool_name] = count
        
        if count >= self.threshold:
            log("circuit", session_id, f"Circuit OPEN for tool '{tool_name}' ({count} failures)", level="warn")
            return True
        return False

    def record_success(self, session_id: str, tool_name: str):
        """Reset the failure count on success."""
        if session_id in self.failures and tool_name in self.failures[session_id]:
            self.failures[session_id][tool_name] = 0

    def is_open(self, session_id: str, tool_name: str) -> bool:
        """Check if the circuit is currently open for a tool."""
        count = self.failures.get(session_id, {}).get(tool_name, 0)
        return count >= self.threshold

    def get_pivot_message(self, tool_name: str) -> str:
        """Standard injection message when a circuit is open."""
        return (
            f"\n\n[SYSTEM: Tool '{tool_name}' failed {self.threshold} times. "
            "Circuit breaker is OPEN. Stop attempting it. "
            "Explain the failure to the user and offer an alternative or proceed without it.]"
        )

    def get_failed_tools(self, session_id: str) -> List[str]:
        """Return tools that have at least one recorded failure in this session."""
        return [
            tool_name
            for tool_name, count in self.failures.get(session_id, {}).items()
            if count > 0
        ]
