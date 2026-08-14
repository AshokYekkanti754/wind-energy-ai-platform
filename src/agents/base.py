"""
The common input/output contract every agent follows, per the work plan's
Day 6 instruction: "Implement each agent as a Python function or class with
a defined input/output contract (so they can be swapped for real ML models
later)." A rule-based GridAgent today and a trained RL GridAgent later both
return the same AgentResult shape, so the pipeline never has to change.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AgentResult:
    agent_name: str
    output: dict[str, Any]
    explanation: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_log_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "agent": self.agent_name,
            "output": self.output,
            "explanation": self.explanation,
        }


class BaseAgent:
    """
    Subclasses implement `run(...)` and must return an AgentResult. `name`
    is used in logs, so give it a short, presentation-ready label.
    """
    name: str = "BaseAgent"

    def run(self, *args, **kwargs) -> AgentResult:
        raise NotImplementedError("Each agent must implement run().")
