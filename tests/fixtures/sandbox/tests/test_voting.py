"""Tests for the voting system."""

from __future__ import annotations

from game.state import GameState
from game.voting import VotingSystem
from game.events import EventBus


def test_voting_round() -> None:
    """A voting round should produce one vote per agent."""
    EventBus.reset()
    state = GameState()
    vs = VotingSystem(state, agent_types=["claude", "gpt"])
    context = {"options": ["cooperate", "defect"]}
    votes = vs.run_vote(context)
    assert len(votes) == 2
    for decision in votes.values():
        assert decision in ("cooperate", "defect")


def test_vote_tally() -> None:
    """Tally should return the most common vote."""
    EventBus.reset()
    state = GameState()
    vs = VotingSystem(state)
    votes = {"a": "cooperate", "b": "cooperate", "c": "defect"}
    result = vs.tally(votes)
    assert result == "cooperate"
