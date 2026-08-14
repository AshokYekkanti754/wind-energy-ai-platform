"""
Simulates the three grid scenarios the work plan calls for (overload,
outage, voltage instability) using synthetic/dummy parameters -- no real
fault data exists for this site. Each scenario returns an AgentResult
(same contract as the Day 6 agents) with a suggested corrective action,
so these slot directly into the same logging pipeline as GridAgent.
"""
from src.agents.base import AgentResult
from src.utils.config_loader import load_config


def simulate_overload_scenario(schedule_summary: dict, cfg: dict | None = None) -> AgentResult:
    """
    Overload: assumes a sudden extra local load spike (e.g. a large
    industrial customer coming online) pushes demand well above the
    day-ahead plan for a few hours -- checks whether the battery pool
    still has enough headroom to absorb it.
    """
    cfg = cfg or load_config()
    spike_kwh = cfg.get("agents", {}).get("grid_export_capacity_kwh_per_day", 22000) * 0.15  # assumed +15% spike

    remaining_import_capacity = schedule_summary["total_demand_kwh"] * 0.3  # ASSUMED grid import limit headroom
    can_absorb = spike_kwh <= remaining_import_capacity

    action = (
        "Sufficient grid import headroom to absorb the spike without shedding load."
        if can_absorb else
        "Grid import headroom insufficient -- recommend emergency battery discharge and/or "
        "controlled load shedding on non-critical circuits."
    )
    output = {
        "scenario": "overload",
        "assumed_spike_kwh": round(spike_kwh, 1),
        "assumed_available_import_headroom_kwh": round(remaining_import_capacity, 1),
        "can_absorb_without_shedding": bool(can_absorb),
        "corrective_action": action,
        "SYNTHETIC_scenario_parameters": True,
    }
    explanation = (
        f"OVERLOAD scenario (synthetic, +15% demand spike assumed = {spike_kwh:,.0f} kWh): {action}"
    )
    return AgentResult(agent_name="Grid Agent (scenario: overload)", output=output, explanation=explanation)


def simulate_outage_scenario(schedule_summary: dict, cfg: dict | None = None,
                              outage_duration_hours: int = 3) -> AgentResult:
    """
    Outage: assumes the grid connection itself drops for N hours (e.g.
    upstream substation fault) -- checks whether the battery pool alone
    can cover local demand for the outage window (island mode).
    """
    cfg = cfg or load_config()
    battery_cfg = cfg.get("battery", {})
    n = battery_cfg.get("n_batteries", 4)
    usable_kwh = (
        battery_cfg.get("capacity_kwh", 2000) * n
        * (battery_cfg.get("max_soc_pct", 95) - battery_cfg.get("min_soc_pct", 10)) / 100
    )
    avg_hourly_demand_kwh = schedule_summary["total_demand_kwh"] / 24
    outage_demand_kwh = avg_hourly_demand_kwh * outage_duration_hours
    can_island = usable_kwh >= outage_demand_kwh

    action = (
        f"Battery pool can island the site for the full {outage_duration_hours}h outage on stored charge alone."
        if can_island else
        f"Battery pool cannot fully cover a {outage_duration_hours}h outage -- recommend load "
        f"shedding to non-critical circuits and prioritizing turbine control/safety systems."
    )
    output = {
        "scenario": "outage",
        "assumed_outage_duration_hours": outage_duration_hours,
        "assumed_outage_demand_kwh": round(outage_demand_kwh, 1),
        "usable_battery_capacity_kwh": round(usable_kwh, 1),
        "can_island_fully": bool(can_island),
        "corrective_action": action,
        "SYNTHETIC_scenario_parameters": True,
    }
    explanation = (
        f"OUTAGE scenario (synthetic, {outage_duration_hours}h grid loss assumed, "
        f"~{outage_demand_kwh:,.0f} kWh demand to cover): {action}"
    )
    return AgentResult(agent_name="Grid Agent (scenario: outage)", output=output, explanation=explanation)


def simulate_voltage_instability_scenario(cfg: dict | None = None,
                                           assumed_deviation_pct: float = 6.0) -> AgentResult:
    """
    Voltage instability: assumes grid voltage has drifted outside the
    typical +/-5% operational band (a common utility tolerance figure, not
    a site-measured one) -- flags whether reactive power support or a
    turbine trip is the appropriate response.
    """
    cfg = cfg or load_config()
    tolerance_pct = 5.0  # commonly used utility voltage tolerance band, ASSUMED here

    within_trip_range = assumed_deviation_pct > tolerance_pct * 1.5  # arbitrary demo trip threshold

    if within_trip_range:
        action = "Deviation exceeds safe operating band -- recommend automatic turbine trip/disconnect to protect equipment."
    else:
        action = "Deviation outside normal tolerance but within safe range -- recommend reactive power (VAR) support to stabilize voltage before considering a trip."

    output = {
        "scenario": "voltage_instability",
        "assumed_voltage_deviation_pct": assumed_deviation_pct,
        "assumed_tolerance_band_pct": tolerance_pct,
        "recommend_trip": bool(within_trip_range),
        "corrective_action": action,
        "SYNTHETIC_scenario_parameters": True,
    }
    explanation = (
        f"VOLTAGE INSTABILITY scenario (synthetic, {assumed_deviation_pct}% deviation vs. "
        f"assumed +/-{tolerance_pct}% tolerance): {action}"
    )
    return AgentResult(agent_name="Grid Agent (scenario: voltage instability)", output=output, explanation=explanation)


def run_all_scenarios(schedule_summary: dict, cfg: dict | None = None) -> list[AgentResult]:
    cfg = cfg or load_config()
    return [
        simulate_overload_scenario(schedule_summary, cfg),
        simulate_outage_scenario(schedule_summary, cfg),
        simulate_voltage_instability_scenario(cfg),
    ]
