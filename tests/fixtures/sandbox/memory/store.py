"""Key-value memory store with indexing support."""

from __future__ import annotations

from typing import Any

from .index import MemoryIndex


class MemoryStore:
    """Simple key-value store backed by a MemoryIndex for search."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._index = MemoryIndex()

    def save(self, key: str, value: Any) -> None:
        """Save a value and update the index."""
        self._data[key] = value
        self._index.add(key, value)

    def load(self, key: str) -> Any | None:
        """Load a value by key."""
        return self._data.get(key)

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if it existed."""
        if key in self._data:
            del self._data[key]
            self._index.remove(key)
            return True
        return False

    def search(self, query: str) -> list[str]:
        """Search for keys matching query via the index."""
        return self._index.search(query)
