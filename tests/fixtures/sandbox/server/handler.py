"""WebSocket handler for multiplayer game communication."""

from __future__ import annotations

from typing import Any

from game.state import GameState
from game.events import get_instance


class WebSocketHandler:
    """Handles incoming WebSocket messages for the game server."""

    def __init__(self, state: GameState) -> None:
        self.state = state
        self._bus = get_instance()
        self._bus.subscribe("vote_complete", self._on_vote_result)

    def on_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Process an incoming message. Returns a response dict."""
        try:
            action = msg["action"]
            if action == "join":
                self.state.save_player({"id": msg["player_id"], "score": 0})
                return {"status": "ok", "action": "joined"}
            if action == "leave":
                self.state.delete_player(msg["player_id"])
                return {"status": "ok", "action": "left"}
            raise ValueError(f"Unknown action: {action!r}")
        except (KeyError, ValueError) as exc:
            return {"status": "error", "detail": str(exc)}

    def handle_reconnect(self, player_id: str) -> dict[str, Any]:
        """Handle a player reconnecting to the server."""
        player = self.state.get_player(player_id)
        if player is not None:
            self._bus.emit("player_reconnect", player)
            return {"status": "ok", "player": player}
        return {"status": "error", "detail": "player not found"}

    def _on_vote_result(self, votes: dict[str, str]) -> None:
        """Internal callback for vote results."""
        self._bus.emit("broadcast", {"type": "vote_result", "votes": votes})
