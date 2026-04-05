"""Tests for agent creation and decision-making."""

from __future__ import annotations

from agents.factory import create_agent
from agents.claude_agent import ClaudeAgent
from agents.gpt_agent import GPTAgent


def test_create_agent() -> None:
    """Factory should create the correct agent subclass."""
    agent = create_agent("claude", name="test-claude")
    assert isinstance(agent, ClaudeAgent)
    assert agent.model == "claude"

    agent2 = create_agent("gpt", name="test-gpt")
    assert isinstance(agent2, GPTAgent)
    assert agent2.model == "gpt"


def test_agent_decide() -> None:
    """Agents should return a decision from the provided options."""
    agent = create_agent("claude", name="decider")
    context = {"options": ["cooperate", "defect"]}
    result = agent.decide(context)
    assert result in ("cooperate", "defect")


def test_agent_repr() -> None:
    """Agent repr should include class name and attributes."""
    agent = create_agent("gpt", name="repr-test")
    text = repr(agent)
    assert "GPTAgent" in text
    assert "repr-test" in text
