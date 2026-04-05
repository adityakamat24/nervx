"""Event bus for decoupled communication between game components."""

from __future__ import annotations

from typing import Any, Callable


class EventBus:
    """Publish-subscribe event bus (singleton via get_instance)."""

    _instance: EventBus | None = None

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable[..., Any]]] = {}

    @classmethod
    def get_instance(cls) -> EventBus:
        """Return the singleton EventBus instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def subscribe(self, event: str, callback: Callable[..., Any]) -> None:
        """Register a callback for an event."""
        self._listeners.setdefault(event, []).append(callback)

    def emit(self, event: str, data: Any = None) -> None:
        """Emit an event, calling all registered callbacks."""
        for callback in self._listeners.get(event, []):
            callback(data)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful in tests)."""
        cls._instance = None


def get_instance() -> EventBus:
    """Module-level convenience for EventBus.get_instance()."""
    return EventBus.get_instance()
