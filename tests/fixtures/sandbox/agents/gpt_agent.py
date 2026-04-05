"""GPT-based agent implementation."""

from __future__ import annotations

from typing import Any

from .base_agent import BaseAgent
from memory.store import MemoryStore


class GPTAgent(BaseAgent):
    """Agent that uses GPT for decision-making."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name, model="gpt")
        self.memory = MemoryStore()

    def decide(self, context: dict[str, Any]) -> str:
        """Use GPT-style reasoning to decide."""
        self.memory.save("last_context", context)
        options = context.get("options", [])
        return options[-1] if options else "pass"

    async def connect(self, ws_manager: Any) -> None:
        """Connect to websocket manager."""
        self._connected = True
