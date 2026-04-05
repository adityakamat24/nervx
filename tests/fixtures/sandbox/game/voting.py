"""Voting system that coordinates agents to vote on game actions."""

from __future__ import annotations

from typing import Any

from agents.factory import create_agent
from game.state import GameState
from game.events import get_instance


class VotingSystem:
    """Manages a voting round among agents."""

    def __init__(self, state: GameState, agent_types: list[str] | None = None) -> None:
        self.state = state
        self.agent_types = agent_types or ["claude", "gpt"]
        self._agents = [
            create_agent(t, name=f"{t}-{i}") for i, t in enumerate(self.agent_types)
        ]

    def run_vote(self, context: dict[str, Any]) -> dict[str, str]:
        """Run a voting round. Each agent decides, results are tallied."""
        votes: dict[str, str] = {}
        for agent in self._agents:
            decision = agent.decide(context)
            votes[agent.name] = decision
        bus = get_instance()
        bus.emit("vote_complete", votes)
        return votes

    def tally(self, votes: dict[str, str]) -> str:
        """Return the most common vote."""
        counts: dict[str, int] = {}
        for vote in votes.values():
            counts[vote] = counts.get(vote, 0) + 1
        return max(counts, key=lambda k: counts[k])
