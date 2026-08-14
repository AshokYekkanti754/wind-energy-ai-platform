"""
Day-ahead battery charge/discharge schedule over a 24-hour horizon, given
hourly generation and demand profiles. Uses PuLP (linear programming) as
the primary method, per the work plan's suggestion; falls back to a greedy
hour-by-hour heuristic if PuLP isn't installed -- the work plan explicitly
allows either ("rule-based or lightweight optimization script").

Models one AGGREGATE battery pool (summed capacity/power across all mock
batteries) for tractability. This is the day-ahead planning layer; the
Optimization Agent (Day 6) still handles fast, single-step, per-battery
real-time decisions -- the two are complementary, not duplicates.
"""
import numpy as np
import pandas as pd


def _aggregate_battery_params(battery_cfg: dict) -> dict:
    n = battery_cfg.get("n_batteries", 4)
    return {
        "capacity_kwh": battery_cfg.get("capacity_kwh", 2000) * n,
        "max_charge_kwh_per_hour": battery_cfg.get("max_charge_rate_kw", 500) * n,
        "max_discharge_kwh_per_hour": battery_cfg.get("max_discharge_rate_kw", 500) * n,
        "min_soc_pct": battery_cfg.get("min_soc_pct", 10),
        "max_soc_pct": battery_cfg.get("max_soc_pct", 95),
        "round_trip_efficiency": battery_cfg.get("round_trip_efficiency", 0.9),
    }


def schedule_battery_lp(generation: pd.Series, demand: pd.Series, battery_cfg: dict,
                         initial_soc_pct: float = 50.0) -> tuple[pd.DataFrame, str]:
    """
    Solves the day-ahead schedule with PuLP. Minimizes total grid import
    over the horizon (equivalent to maximizing self-consumption / minimizing
    reliance on the grid) subject to battery physics constraints.

    Returns (schedule_df, backend) where backend is "pulp" or "heuristic"
    (falls back automatically if PuLP or its bundled CBC solver isn't usable).
    """
    try:
        import pulp
        return _schedule_lp(generation, demand, battery_cfg, initial_soc_pct), "pulp"
    except Exception:
        return _schedule_heuristic(generation, demand, battery_cfg, initial_soc_pct), "heuristic"


def _schedule_lp(generation: pd.Series, demand: pd.Series, battery_cfg: dict,
                  initial_soc_pct: float) -> pd.DataFrame:
    import pulp

    params = _aggregate_battery_params(battery_cfg)
    hours = list(range(24))
    eff = params["round_trip_efficiency"]
    cap = params["capacity_kwh"]
    soc_min = params["min_soc_pct"] / 100 * cap
    soc_max = params["max_soc_pct"] / 100 * cap
    initial_soc = initial_soc_pct / 100 * cap

    prob = pulp.LpProblem("battery_schedule", pulp.LpMinimize)

    charge = pulp.LpVariable.dicts("charge", hours, lowBound=0, upBound=params["max_charge_kwh_per_hour"])
    discharge = pulp.LpVariable.dicts("discharge", hours, lowBound=0, upBound=params["max_discharge_kwh_per_hour"])
    grid_import = pulp.LpVariable.dicts("grid_import", hours, lowBound=0)
    grid_export = pulp.LpVariable.dicts("grid_export", hours, lowBound=0)
    soc = pulp.LpVariable.dicts("soc", hours, lowBound=soc_min, upBound=soc_max)

    # Objective: minimize grid import (small penalty on export too, so the
    # battery isn't indifferent between exporting and charging when both
    # are equally valid -- slight preference for self-consumption via storage)
    prob += pulp.lpSum(grid_import[t] for t in hours) + 0.01 * pulp.lpSum(grid_export[t] for t in hours)

    for t in hours:
        gen_t = float(generation.iloc[t])
        dem_t = float(demand.iloc[t])
        # energy balance: sources == uses
        prob += gen_t + discharge[t] + grid_import[t] == dem_t + charge[t] + grid_export[t]

        prev_soc = initial_soc if t == 0 else soc[t - 1]
        prob += soc[t] == prev_soc + charge[t] * eff - discharge[t] / eff

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    rows = []
    for t in hours:
        rows.append({
            "hour": t,
            "generation_kwh": float(generation.iloc[t]),
            "demand_kwh": float(demand.iloc[t]),
            "charge_kwh": charge[t].value(),
            "discharge_kwh": discharge[t].value(),
            "grid_import_kwh": grid_import[t].value(),
            "grid_export_kwh": grid_export[t].value(),
            "soc_kwh": soc[t].value(),
            "soc_pct": round(soc[t].value() / cap * 100, 1),
        })
    return pd.DataFrame(rows)


def _schedule_heuristic(generation: pd.Series, demand: pd.Series, battery_cfg: dict,
                          initial_soc_pct: float) -> pd.DataFrame:
    """
    Greedy hour-by-hour rule: charge from surplus generation, discharge to
    cover deficits, both capped by rate limits and SOC bounds. Not globally
    optimal (a real LP can, e.g., hold charge back in hour 6 to cover a
    bigger deficit in hour 18), but transparent and dependency-free.
    """
    params = _aggregate_battery_params(battery_cfg)
    cap = params["capacity_kwh"]
    eff = params["round_trip_efficiency"]
    soc_min = params["min_soc_pct"] / 100 * cap
    soc_max = params["max_soc_pct"] / 100 * cap
    soc = initial_soc_pct / 100 * cap

    rows = []
    for t in range(24):
        gen_t = float(generation.iloc[t])
        dem_t = float(demand.iloc[t])
        balance = gen_t - dem_t

        charge_kwh = discharge_kwh = 0.0
        if balance > 0:
            room_kwh = (soc_max - soc) / eff
            charge_kwh = min(balance, params["max_charge_kwh_per_hour"], room_kwh)
            soc += charge_kwh * eff
        else:
            available_kwh = (soc - soc_min) * eff
            discharge_kwh = min(-balance, params["max_discharge_kwh_per_hour"], available_kwh)
            soc -= discharge_kwh / eff

        net_after_battery = balance - charge_kwh + discharge_kwh
        grid_import_kwh = max(-net_after_battery, 0.0)
        grid_export_kwh = max(net_after_battery, 0.0)

        rows.append({
            "hour": t, "generation_kwh": gen_t, "demand_kwh": dem_t,
            "charge_kwh": round(charge_kwh, 2), "discharge_kwh": round(discharge_kwh, 2),
            "grid_import_kwh": round(grid_import_kwh, 2), "grid_export_kwh": round(grid_export_kwh, 2),
            "soc_kwh": round(soc, 2), "soc_pct": round(soc / cap * 100, 1),
        })
    return pd.DataFrame(rows)


def summarize_schedule(schedule: pd.DataFrame) -> dict:
    total_generation = schedule["generation_kwh"].sum()
    total_demand = schedule["demand_kwh"].sum()
    total_import = schedule["grid_import_kwh"].sum()
    total_export = schedule["grid_export_kwh"].sum()
    self_sufficiency_pct = round((1 - total_import / total_demand) * 100, 1) if total_demand else None

    return {
        "total_generation_kwh": round(total_generation, 1),
        "total_demand_kwh": round(total_demand, 1),
        "total_grid_import_kwh": round(total_import, 1),
        "total_grid_export_kwh": round(total_export, 1),
        "self_sufficiency_pct": self_sufficiency_pct,
        "final_soc_pct": float(schedule["soc_pct"].iloc[-1]),
    }
