"""Base agent with abstract interface for all AI agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base class for all game agents."""

    def __init__(self, name: str, model: str) -> None:
        self.name = name
        self.model = model
        self._connected = False

    @abstractmethod
    def decide(self, context: dict[str, Any]) -> str:
        """Return a decision string given game context."""

    @abstractmethod
    async def connect(self, ws_manager: Any) -> None:
        """Connect to a websocket manager."""

    @property
    def is_connected(self) -> bool:
        """Whether the agent is currently connected."""
        return self._connected

    @staticmethod
    def supported_models() -> list[str]:
        """Return list of supported model identifiers."""
        return ["claude", "gpt"]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model!r})"
