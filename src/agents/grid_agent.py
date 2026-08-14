"""
Grid Agent -- checks the Forecast Agent's output against a grid export
capacity limit and flags overload/curtailment risk. This is the Day 6
single-check version; Day 7 (src/optimization/grid_scenarios.py) adds the
three fuller scenario simulations (overload, outage, voltage instability)
the work plan calls for.

The capacity figure is an ASSUMED planning constraint (config.yaml,
agents.grid_export_capacity_kwh_per_day) -- Dataset-9's real grid data
doesn't overlap the SCADA period (confirmed Day 2), so there's no measured
limit to check against here. State this plainly if asked in the demo.
"""
from src.agents.base import BaseAgent, AgentResult
from src.utils.config_loader import load_config


class GridAgent(BaseAgent):
    name = "Grid Agent"

    def run(self, forecast_result: AgentResult, cfg: dict | None = None) -> AgentResult:
        cfg = cfg or load_config()
        capacity_kwh = cfg.get("agents", {}).get("grid_export_capacity_kwh_per_day", 22000)
        forecast_kwh = forecast_result.output["next_day_generation_forecast_kwh"]

        headroom_kwh = capacity_kwh - forecast_kwh
        overload_risk = headroom_kwh < 0

        if overload_risk:
            corrective_action = (
                f"Forecast exceeds assumed export capacity by {abs(headroom_kwh):,.0f} kWh -- "
                f"recommend routing surplus to battery storage or accepting curtailment of that amount."
            )
        elif headroom_kwh < capacity_kwh * 0.1:
            corrective_action = "Within capacity but headroom is thin (<10%) -- monitor closely tomorrow."
        else:
            corrective_action = "No corrective action needed -- forecast is comfortably within assumed export capacity."

        output = {
            "assumed_export_capacity_kwh": capacity_kwh,
            "forecast_generation_kwh": forecast_kwh,
            "headroom_kwh": round(float(headroom_kwh), 1),
            "overload_risk": bool(overload_risk),
            "corrective_action": corrective_action,
            "ASSUMED_capacity_not_measured": True,
        }
        explanation = (
            f"Tomorrow's forecast ({forecast_kwh:,.0f} kWh) vs. assumed export capacity "
            f"({capacity_kwh:,.0f} kWh): {'OVERLOAD RISK' if overload_risk else 'within capacity'}, "
            f"headroom {headroom_kwh:,.0f} kWh. {corrective_action} "
            f"[Capacity figure is an assumed planning constraint, not a measured grid limit.]"
        )
        return AgentResult(agent_name=self.name, output=output, explanation=explanation)
