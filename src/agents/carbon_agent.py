"""
Carbon Agent -- computes avoided CO2 and renewable utilization %, layered
on top of the Forecast Agent's output, per the work plan's ESG-tracking
instruction. Uses an assumed grid emission factor (config.yaml,
agents.grid_emission_factor_kg_co2_per_kwh) since no site-specific factor
was provided -- state this is an adjustable assumption, not a measured
figure, if asked in the demo.
"""
from src.agents.base import BaseAgent, AgentResult
from src.utils.config_loader import load_config


class CarbonAgent(BaseAgent):
    name = "Carbon Agent"

    def run(self, forecast_result: AgentResult, cfg: dict | None = None) -> AgentResult:
        cfg = cfg or load_config()
        emission_factor = cfg.get("agents", {}).get("grid_emission_factor_kg_co2_per_kwh", 0.82)

        gen = forecast_result.output
        generation_kwh = gen["next_day_generation_forecast_kwh"]
        demand_kwh = gen.get("next_day_demand_forecast_kwh")

        avoided_co2_kg = generation_kwh * emission_factor
        renewable_utilization_pct = (
            round(min(generation_kwh / demand_kwh, 1.0) * 100, 1) if demand_kwh else None
        )

        output = {
            "forecast_generation_kwh": generation_kwh,
            "assumed_grid_emission_factor_kg_per_kwh": emission_factor,
            "avoided_co2_kg": round(avoided_co2_kg, 1),
            "avoided_co2_tonnes": round(avoided_co2_kg / 1000, 3),
            "renewable_utilization_pct": renewable_utilization_pct,
            "ASSUMPTION_note": "emission factor is a documented assumption, not a measured site-specific figure",
        }
        util_phrase = (
            f"covering an estimated {renewable_utilization_pct}% of forecasted demand (proxy)"
            if renewable_utilization_pct is not None else "demand forecast unavailable for a utilization estimate"
        )
        explanation = (
            f"Tomorrow's {generation_kwh:,.0f} kWh forecast avoids an estimated "
            f"{avoided_co2_kg:,.0f} kg CO2 (~{avoided_co2_kg/1000:.2f} t), using an assumed grid "
            f"emission factor of {emission_factor} kg/kWh, {util_phrase}. "
            f"[Emission factor is an adjustable assumption, not measured for this site.]"
        )
        return AgentResult(agent_name=self.name, output=output, explanation=explanation)
