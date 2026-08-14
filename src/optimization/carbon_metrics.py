"""
Layers carbon/ESG metrics on top of the battery schedule's actual hourly
numbers -- more granular than Day 6's CarbonAgent, which works from the
single daily forecast total. Same assumed emission factor (config.yaml),
same "this is an assumption, not a measured figure" caveat applies.
"""
import pandas as pd


def compute_schedule_carbon_metrics(schedule: pd.DataFrame, cfg: dict) -> dict:
    emission_factor = cfg.get("agents", {}).get("grid_emission_factor_kg_co2_per_kwh", 0.82)

    total_generation_kwh = schedule["generation_kwh"].sum()
    total_demand_kwh = schedule["demand_kwh"].sum()
    total_import_kwh = schedule["grid_import_kwh"].sum()

    # renewable-covered demand = demand met by wind generation or battery
    # (battery is charged from wind here, so its discharge counts as renewable too)
    renewable_covered_kwh = total_demand_kwh - total_import_kwh
    renewable_utilization_pct = round(renewable_covered_kwh / total_demand_kwh * 100, 1) if total_demand_kwh else None

    avoided_co2_kg = total_generation_kwh * emission_factor
    grid_import_co2_kg = total_import_kwh * emission_factor  # CO2 attributable to grid-imported energy

    return {
        "total_generation_kwh": round(total_generation_kwh, 1),
        "total_demand_kwh": round(total_demand_kwh, 1),
        "renewable_covered_demand_kwh": round(renewable_covered_kwh, 1),
        "renewable_utilization_pct": renewable_utilization_pct,
        "avoided_co2_kg": round(avoided_co2_kg, 1),
        "avoided_co2_tonnes": round(avoided_co2_kg / 1000, 3),
        "grid_import_attributed_co2_kg": round(grid_import_co2_kg, 1),
        "assumed_emission_factor_kg_per_kwh": emission_factor,
    }
