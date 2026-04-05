"""Claude-based agent implementation."""

from __future__ import annotations

from typing import Any

from .base_agent import BaseAgent
from memory.store import MemoryStore


class ClaudeAgent(BaseAgent):
    """Agent that uses Claude for decision-making."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name, model="claude")
        self.memory = MemoryStore()

    def decide(self, context: dict[str, Any]) -> str:
        """Use Claude-style reasoning to decide."""
        self.memory.save("last_context", context)
        options = context.get("options", [])
        return options[0] if options else "abstain"

    async def connect(self, ws_manager: Any) -> None:
        """Connect to websocket manager."""
        self._connected = True

    def reflect(self, outcome: str) -> None:
        """Store reflection on past outcome."""
        self.memory.save("reflection", outcome)
