"""
Wires the five agents into the pipeline the work plan describes:
Forecast Agent -> Optimization Agent consumes predictions -> Grid Agent
checks for overload -> (Maintenance Agent and Carbon Agent run alongside,
both also consuming the Forecast Agent's output where relevant).

Every agent's decision is logged as one JSON line per run to
logs/agent_decisions_<date>.jsonl -- this is the "explainability" log the
work plan calls for, and it's what Day 8's dashboard reads to show the
agent activity feed.
"""
import json
from datetime import datetime
from pathlib import Path

from src.agents.base import AgentResult
from src.agents.forecast_agent import ForecastAgent
from src.agents.optimization_agent import OptimizationAgent
from src.agents.maintenance_agent import MaintenanceAgent
from src.agents.grid_agent import GridAgent
from src.agents.carbon_agent import CarbonAgent
from src.utils.config_loader import load_config


def run_pipeline(cfg: dict | None = None) -> list[AgentResult]:
    """Runs all five agents in dependency order and returns their results, in run order."""
    cfg = cfg or load_config()

    forecast_result = ForecastAgent().run(cfg)
    maintenance_result = MaintenanceAgent().run(cfg)
    optimization_result = OptimizationAgent().run(forecast_result, cfg)
    grid_result = GridAgent().run(forecast_result, cfg)
    carbon_result = CarbonAgent().run(forecast_result, cfg)

    return [forecast_result, maintenance_result, optimization_result, grid_result, carbon_result]


def log_pipeline_run(results: list[AgentResult], cfg: dict | None = None) -> Path:
    """Appends each agent's decision as one JSON line to a per-day log file."""
    cfg = cfg or load_config()
    logs_dir = Path(cfg["paths"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"agent_decisions_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(log_path, "a") as f:
        for result in results:
            f.write(json.dumps(result.to_log_dict(), default=str) + "\n")
    return log_path


def print_pipeline_summary(results: list[AgentResult]) -> None:
    """Presentation-ready console summary -- one block per agent."""
    for result in results:
        print(f"[{result.agent_name}]")
        print(f"  {result.explanation}")
        print()
