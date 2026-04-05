"""Factory for creating agents by type string."""

from __future__ import annotations

from .base_agent import BaseAgent
from .claude_agent import ClaudeAgent
from .gpt_agent import GPTAgent


_AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "claude": ClaudeAgent,
    "gpt": GPTAgent,
}


def create_agent(agent_type: str, name: str = "default") -> BaseAgent:
    """Create an agent instance by type identifier.

    Args:
        agent_type: One of 'claude' or 'gpt'.
        name: Display name for the agent.

    Returns:
        A BaseAgent subclass instance.

    Raises:
        ValueError: If agent_type is not recognized.
    """
    cls = _AGENT_REGISTRY.get(agent_type)
    if cls is None:
        raise ValueError(f"Unknown agent type: {agent_type!r}")
    return cls(name=name)
