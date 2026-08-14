"""
Forecast Agent -- produces the next-day generation and demand forecasts
that every downstream agent consumes. Thin wrapper around
context_builder.get_generation_context so the Day 3/4 models can be swapped
for something else later without touching this file's callers.
"""
from src.agents.base import BaseAgent, AgentResult
from src.agents.context_builder import get_generation_context
from src.utils.config_loader import load_config


class ForecastAgent(BaseAgent):
    name = "Forecast Agent"

    def run(self, cfg: dict | None = None) -> AgentResult:
        cfg = cfg or load_config()
        gen = get_generation_context(cfg)

        trend = gen.get("generation_vs_trailing_avg_pct")
        trend_phrase = (
            f"{'up' if trend and trend > 0 else 'down'} {abs(trend):.1f}% vs. the trailing 7-day average"
            if trend is not None else "no trailing comparison available"
        )

        explanation = (
            f"Forecasting {gen['next_day_generation_forecast_kwh']:,.0f} kWh generation for the day "
            f"after {gen['anchor_date']} (model test MAE: {gen['model_test_mae_kwh']:,.0f} kWh). "
            f"Today's actual generation was {gen['actual_generation_today_kwh']:,.0f} kWh, {trend_phrase}. "
        )
        if gen.get("next_day_demand_forecast_kwh"):
            explanation += (
                f"Demand proxy (SYNTHETIC, not real load) forecast: "
                f"{gen['next_day_demand_forecast_kwh']:,.0f} kWh."
            )

        return AgentResult(agent_name=self.name, output=gen, explanation=explanation)
