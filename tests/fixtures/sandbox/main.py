"""Entrypoint for the multiplayer game."""

from __future__ import annotations

from agents.factory import create_agent
from agents.base_agent import BaseAgent
from game.state import GameState
from game.voting import VotingSystem
from game.events import EventBus, get_instance
from memory.store import MemoryStore
from server.handler import WebSocketHandler


def main() -> None:
    """Run a demo game round."""
    # Set up state and event bus
    state = GameState()
    bus = get_instance()

    log: list[str] = []
    bus.subscribe("vote_complete", lambda v: log.append(f"votes: {v}"))

    # Add players
    state.save_player({"id": "p1", "score": 0})
    state.save_player({"id": "p2", "score": 0})

    # Create agents via factory
    agents: list[BaseAgent] = [
        create_agent("claude", name="agent-1"),
        create_agent("gpt", name="agent-2"),
    ]

    # Run a voting round
    voting = VotingSystem(state, agent_types=["claude", "gpt"])
    context = {"options": ["cooperate", "defect"]}
    votes = voting.run_vote(context)
    winner = voting.tally(votes)

    # Update scores
    state.update_score("p1", 10)

    # Server handler demo
    handler = WebSocketHandler(state)
    response = handler.on_message({"action": "join", "player_id": "p3"})

    # Memory demo
    store = MemoryStore()
    store.save("round_1", {"votes": votes, "winner": winner})

    print(f"Players: {state.player_count}")
    print(f"Winner: {winner}")
    print(f"Server response: {response}")
    print(f"Log: {log}")


if __name__ == "__main__":
    main()
