"""
The GenAI Energy Copilot. Feeds the structured context from context_builder
into an LLM (Groq API) alongside the operator's question.

Per the work plan's Day 9 instruction ("fallback/error handling so the demo
doesn't break if an API call fails or data is missing"): if no API key is
set, or the API call fails for any reason, this falls back to a rule-based
templated answer built directly from the context dict. The fallback is
less fluent but never leaves the demo silent.
"""
import json
import os


SYSTEM_PROMPT_TEMPLATE = """You are the Energy Copilot for an AI-powered wind farm operations platform.
Answer the operator's question using ONLY the data context provided below. Be concise and specific,
citing actual numbers from the context. If something isn't covered by the context, say so plainly
rather than guessing.

IMPORTANT: any data below marked MOCKED is placeholder data for this demo, not a real reading.
If your answer relies on mocked data, say so in the answer (e.g. "based on mocked battery data...").

DATA CONTEXT:
{context_json}
"""


def build_system_prompt(context: dict) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(context_json=json.dumps(context, indent=2, default=str))


def rule_based_fallback(question: str, context: dict) -> str:
    """
    A deliberately simple templated answer, used only when the LLM call
    can't be made. Covers the four sample questions from the work plan
    directly; anything else gets an honest "can't answer without the API."
    """
    q = question.lower()
    gen = context["generation"]

    if "drop" in q or "why did" in q:
        trend = gen.get("generation_vs_trailing_avg_pct")
        if trend is None:
            trend_note = "No trailing-average comparison is available yet."
        elif trend < -5:
            trend_note = f"Generation was actually down {abs(trend):.1f}% vs. the trailing 7-day average."
        elif trend > 5:
            trend_note = f"Generation was actually UP {trend:.1f}% vs. the trailing 7-day average -- not a drop."
        else:
            trend_note = f"Generation was roughly in line with the trailing 7-day average ({trend:+.1f}%)."
        return (
            f"On {gen['anchor_date']}, WTG_001 generated {gen['actual_generation_today_kwh']:,.0f} kWh "
            f"(trailing 7-day average: {gen.get('trailing_7d_avg_generation_kwh', 0):,.0f} kWh). {trend_note} "
            f"Estimated power loss vs. the expected power curve was {gen['actual_power_loss_today_kwh']:,.0f} kWh. "
            f"Mean wind speed was {gen['mean_wind_speed_today_mps']} m/s, "
            f"{gen['wind_vs_seasonal_pct']:+.1f}% vs. the seasonal average for this week. "
            f"There were {gen['alarm_minutes_today']} minutes of active alarms that day. "
            f"[Rule-based fallback answer -- set GROQ_API_KEY for a fuller explanation.]"
        )

    if "predict" in q or "tomorrow" in q or "forecast" in q:
        demand_note = (
            f" Demand proxy forecast (MOCKED/synthetic, not real load): {gen['next_day_demand_forecast_kwh']:,.0f} kWh."
            if gen.get("next_day_demand_forecast_kwh") else ""
        )
        return (
            f"Forecast for the day after {gen['anchor_date']}: {gen['next_day_generation_forecast_kwh']:,.0f} kWh "
            f"generation (model test MAE: {gen['model_test_mae_kwh']:,.0f} kWh).{demand_note} "
            f"[Rule-based fallback answer.]"
        )

    if "maintenance" in q or "turbine" in q and "need" in q:
        top = context["turbine_health_top5"][0]
        return (
            f"Highest-risk turbine over the last 90 days: {top['turbine_id']} "
            f"({top['n_failures']} failures, {top['total_downtime_hours']:.0f}h downtime, "
            f"most affected component: {top['most_common_component']}). "
            f"This is a rule-based risk score from real alarm/maintenance history, not a trained "
            f"failure-probability model. [Rule-based fallback answer.]"
        )

    if "battery" in q or "discharge" in q:
        top = context["battery_status_MOCKED"][0]
        return (
            f"[MOCKED DATA -- no real battery dataset exists] Highest dispatch priority: "
            f"{top['battery_id']} (SOC {top['state_of_charge_pct']}%, health {top['health_pct']}%). "
            f"[Rule-based fallback answer.]"
        )

    return (
        "I can't answer that without a live LLM call (no GROQ_API_KEY set, or the API call failed). "
        "The rule-based fallback only covers the four sample operator questions from the work plan."
    )


def ask_copilot(question: str, context: dict, model: str = "llama-3.3-70b-versatile",
                 max_tokens: int = 500) -> dict:
    """
    Returns {"answer": str, "source": "llm" | "fallback", "error": str | None}.
    Never raises -- always falls back to a templated answer on any failure,
    per the work plan's fault-tolerance requirement.

    Uses the Groq API (OpenAI-compatible chat completions). Groq's currently
    recommended general-purpose model is "llama-3.3-70b-versatile"; check
    https://console.groq.com/docs/models if this notebook errors with an
    "model not found" style message, since hosted model names/availability
    can change and aren't something this project pins reliably.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"answer": rule_based_fallback(question, context), "source": "fallback",
                "error": "GROQ_API_KEY not set"}

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": build_system_prompt(context)},
                {"role": "user", "content": question},
            ],
        )
        answer = response.choices[0].message.content
        return {"answer": answer, "source": "llm", "error": None}
    except Exception as e:
        # Deliberately broad: any API failure (bad key, rate limit, network,
        # wrong model string) should degrade to the fallback, not crash the demo.
        return {"answer": rule_based_fallback(question, context), "source": "fallback", "error": str(e)}
