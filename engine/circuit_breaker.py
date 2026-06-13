"""
Tool Circuit Breaker — hardened
-------------------------------
Prevents infinite loops and wasted turns by tracking tool failures.

Hardening over the original:
  - HALF-OPEN state: after a cooldown (measured in turns), an open circuit
    allows ONE trial call. Success closes it; failure re-opens it. A tool
    that failed due to a transient blip is no longer dead for the whole
    session.
  - LRU eviction: per-session state is capped so long-lived processes do
    not grow unbounded.
  - Persistence hooks: optional load/dump so breaker state can survive a
    restart alongside the session, instead of silently resetting to closed.

State machine:  CLOSED ──(threshold failures)──▶ OPEN
                OPEN   ──(cooldown elapsed)────▶ HALF_OPEN
                HALF_OPEN ──(success)──────────▶ CLOSED
                HALF_OPEN ──(failure)──────────▶ OPEN
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from config.logger import log
except Exception:  # pragma: no cover - logging is optional in isolation
    def log(*args: Any, **kwargs: Any) -> None:  # type: ignore
        pass


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _ToolCircuit:
    failures: int = 0
    state: CircuitState = CircuitState.CLOSED
    opened_at_turn: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "failures": self.failures,
            "state": self.state.value,
            "opened_at_turn": self.opened_at_turn,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_ToolCircuit":
        return cls(
            failures=int(data.get("failures") or 0),
            state=CircuitState(str(data.get("state") or "closed")),
            opened_at_turn=data.get("opened_at_turn"),
        )


class CircuitBreaker:
    def __init__(
        self,
        threshold: int = 3,
        *,
        cooldown_turns: int = 3,
        max_sessions: int = 512,
    ):
        self.threshold = threshold
        self.cooldown_turns = cooldown_turns
        self.max_sessions = max_sessions
        # OrderedDict gives us cheap LRU semantics via move_to_end / popitem.
        self._sessions: "OrderedDict[str, Dict[str, _ToolCircuit]]" = OrderedDict()

    # ── internal helpers ─────────────────────────────────────────────
    def _touch(self, session_id: str) -> Dict[str, _ToolCircuit]:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]
        circuits: Dict[str, _ToolCircuit] = {}
        self._sessions[session_id] = circuits
        if len(self._sessions) > self.max_sessions:
            evicted_sid, _ = self._sessions.popitem(last=False)
            log("circuit", evicted_sid, "Evicted breaker state (LRU cap)", level="info")
        return circuits

    def _circuit(self, session_id: str, tool_name: str) -> _ToolCircuit:
        circuits = self._touch(session_id)
        if tool_name not in circuits:
            circuits[tool_name] = _ToolCircuit()
        return circuits[tool_name]

    # ── transitions ──────────────────────────────────────────────────
    def record_failure(self, session_id: str, tool_name: str, *, current_turn: Optional[int] = None) -> bool:
        """Record a failure. Returns True if the circuit is now OPEN."""
        circuit = self._circuit(session_id, tool_name)
        circuit.failures += 1

        # A failure in HALF_OPEN immediately re-opens.
        if circuit.state == CircuitState.HALF_OPEN:
            circuit.state = CircuitState.OPEN
            circuit.opened_at_turn = current_turn
            log("circuit", session_id, f"Circuit RE-OPEN for '{tool_name}' (half-open trial failed)", level="warn")
            return True

        if circuit.failures >= self.threshold and circuit.state != CircuitState.OPEN:
            circuit.state = CircuitState.OPEN
            circuit.opened_at_turn = current_turn
            log("circuit", session_id, f"Circuit OPEN for '{tool_name}' ({circuit.failures} failures)", level="warn")
            return True
        return circuit.state == CircuitState.OPEN

    def record_success(self, session_id: str, tool_name: str) -> None:
        circuit = self._circuit(session_id, tool_name)
        circuit.failures = 0
        circuit.state = CircuitState.CLOSED
        circuit.opened_at_turn = None

    def is_open(self, session_id: str, tool_name: str, *, current_turn: Optional[int] = None) -> bool:
        """Return True if the tool is blocked right now.

        Transitions OPEN → HALF_OPEN when the cooldown has elapsed, allowing
        a single trial call (so this returns False for the trial).
        """
        circuits = self._sessions.get(session_id)
        if not circuits or tool_name not in circuits:
            return False
        circuit = circuits[tool_name]

        if circuit.state == CircuitState.CLOSED:
            return False
        if circuit.state == CircuitState.HALF_OPEN:
            return False  # allow the trial call through

        # state == OPEN: check cooldown
        if (
            current_turn is not None
            and circuit.opened_at_turn is not None
            and (current_turn - circuit.opened_at_turn) >= self.cooldown_turns
        ):
            circuit.state = CircuitState.HALF_OPEN
            log("circuit", session_id, f"Circuit HALF-OPEN for '{tool_name}' (cooldown elapsed, trial allowed)", level="info")
            return False
        return True

    def state_of(self, session_id: str, tool_name: str) -> CircuitState:
        circuits = self._sessions.get(session_id)
        if not circuits or tool_name not in circuits:
            return CircuitState.CLOSED
        return circuits[tool_name].state

    def get_pivot_message(self, tool_name: str) -> str:
        return (
            f"\n\n[SYSTEM: Tool '{tool_name}' failed {self.threshold} times. "
            "Circuit breaker is OPEN. Stop attempting it for now. "
            "Explain the failure to the user and offer an alternative or proceed without it. "
            f"It may be retried once after {self.cooldown_turns} turns.]"
        )

    def get_failed_tools(self, session_id: str) -> List[str]:
        circuits = self._sessions.get(session_id, {})
        return [name for name, c in circuits.items() if c.failures > 0]

    # ── persistence ──────────────────────────────────────────────────
    def dump_session(self, session_id: str) -> dict[str, Any]:
        """Serialize one session's breaker state for storage alongside the session."""
        circuits = self._sessions.get(session_id, {})
        return {name: c.to_dict() for name, c in circuits.items()}

    def load_session(self, session_id: str, data: dict[str, Any]) -> None:
        """Restore breaker state after a restart."""
        if not data:
            return
        circuits = self._touch(session_id)
        for name, raw in data.items():
            try:
                circuits[name] = _ToolCircuit.from_dict(raw)
            except Exception:
                continue

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)