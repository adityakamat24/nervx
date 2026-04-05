"""Index for fast lookups into the memory store."""

from __future__ import annotations

from typing import Any


class MemoryIndex:
    """Maintains a keyword-to-keys index for MemoryStore."""

    def __init__(self) -> None:
        self._index: dict[str, set[str]] = {}

    def add(self, key: str, value: Any) -> None:
        """Index a key by the words in its string representation."""
        for token in str(value).lower().split():
            self._index.setdefault(token, set()).add(key)

    def search(self, query: str) -> list[str]:
        """Return keys matching the query token."""
        return sorted(self._index.get(query.lower(), set()))

    def remove(self, key: str) -> None:
        """Remove a key from all index entries."""
        for token_keys in self._index.values():
            token_keys.discard(key)
