"""
Optimization Agent -- consumes the Forecast Agent's output and suggests a
battery charge/discharge action for tomorrow. Rule-based for the Day 6 demo
(surplus generation -> charge the emptiest healthy battery; deficit ->
discharge the highest-priority battery). Day 7 replaces the single-step
version of this decision with a proper multi-period LP schedule
(src/optimization/battery_scheduler.py) -- this agent stays as the fast,
explainable single-decision version the copilot/dashboard can call cheaply.
"""
from src.agents.base import BaseAgent, AgentResult
from src.agents.context_builder import get_mock_battery_status
from src.utils.config_loader import load_config


class OptimizationAgent(BaseAgent):
    name = "Optimization Agent"

    def run(self, forecast_result: AgentResult, cfg: dict | None = None) -> AgentResult:
        cfg = cfg or load_config()
        gen = forecast_result.output

        generation_kwh = gen["next_day_generation_forecast_kwh"]
        demand_kwh = gen.get("next_day_demand_forecast_kwh")
        battery_cfg = cfg.get("battery", {})
        max_rate_kwh = max(battery_cfg.get("max_charge_rate_kw", 500),
                            battery_cfg.get("max_discharge_rate_kw", 500))

        batteries = get_mock_battery_status(n_batteries=battery_cfg.get("n_batteries", 4))

        if demand_kwh is None:
            output = {"action": "hold", "reason": "no demand forecast available"}
            explanation = "No demand forecast available -- Optimization Agent has nothing to balance against, holding all batteries."
            return AgentResult(agent_name=self.name, output=output, explanation=explanation)

        balance_kwh = generation_kwh - demand_kwh  # positive = surplus, negative = deficit

        if balance_kwh > 0:
            # surplus: charge the battery with the most headroom (lowest SOC) among healthy packs
            candidates = batteries[batteries["health_pct"] >= 85].sort_values("state_of_charge_pct")
            target = candidates.iloc[0] if len(candidates) else batteries.iloc[-1]
            amount_kwh = round(min(balance_kwh, max_rate_kwh), 1)
            action = "charge"
        else:
            # deficit: discharge the battery with the best SOC x health dispatch priority
            target = batteries.iloc[0]  # already sorted by dispatch_priority_score descending
            amount_kwh = round(min(abs(balance_kwh), max_rate_kwh), 1)
            action = "discharge"

        output = {
            "action": action,
            "battery_id": target["battery_id"],
            "amount_kwh": amount_kwh,
            "battery_soc_pct": float(target["state_of_charge_pct"]),
            "battery_health_pct": float(target["health_pct"]),
            "forecast_balance_kwh": round(float(balance_kwh), 1),
            "MOCKED_battery_data": True,
        }
        explanation = (
            f"Forecast shows a {'surplus' if balance_kwh > 0 else 'deficit'} of "
            f"{abs(balance_kwh):,.0f} kWh (generation {generation_kwh:,.0f} vs. demand-proxy "
            f"{demand_kwh:,.0f}). Recommending {action} of {amount_kwh:,.0f} kWh on "
            f"{target['battery_id']} (SOC {target['state_of_charge_pct']}%, health {target['health_pct']}%). "
            f"[Battery data is MOCKED -- no real BESS dataset exists.]"
        )
        return AgentResult(agent_name=self.name, output=output, explanation=explanation)
