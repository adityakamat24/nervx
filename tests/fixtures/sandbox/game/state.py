"""Game state repository - CRUD operations for players."""

from __future__ import annotations

from typing import Any


class GameState:
    """Repository for game state, managing player data."""

    def __init__(self) -> None:
        self._players: dict[str, dict[str, Any]] = {}

    def get_player(self, player_id: str) -> dict[str, Any] | None:
        """Retrieve a player by ID."""
        return self._players.get(player_id)

    def save_player(self, player: dict[str, Any]) -> None:
        """Save or update a player record."""
        player_id = player["id"]
        self._players[player_id] = player

    def update_score(self, player_id: str, score: int) -> None:
        """Update score for an existing player."""
        if player_id in self._players:
            self._players[player_id]["score"] = score

    def delete_player(self, player_id: str) -> bool:
        """Remove a player. Returns True if found and removed."""
        return self._players.pop(player_id, None) is not None

    @property
    def player_count(self) -> int:
        """Number of players currently in the game."""
        return len(self._players)

    def all_player_ids(self) -> list[str]:
        """Return all current player IDs."""
        return list(self._players.keys())
