"""
Maintenance Agent -- flags the highest-risk turbine using the real
alarm/maintenance risk ranking from context_builder (genuine fleet-wide
data, see that module's docstring). Rule-based threshold, not a trained
failure-probability model -- Report Model 4 territory for a future phase.
"""
from src.agents.base import BaseAgent, AgentResult
from src.agents.context_builder import get_turbine_health_ranking
from src.utils.config_loader import load_config


class MaintenanceAgent(BaseAgent):
    name = "Maintenance Agent"

    def run(self, cfg: dict | None = None) -> AgentResult:
        cfg = cfg or load_config()
        threshold = cfg.get("agents", {}).get("maintenance_risk_score_threshold", 15)

        ranking = get_turbine_health_ranking(cfg, top_n=5)
        top = ranking.iloc[0]
        needs_inspection = top["risk_score"] >= threshold

        output = {
            "turbine_id": top["turbine_id"],
            "risk_score": round(float(top["risk_score"]), 1),
            "n_failures_90d": int(top["n_failures"]),
            "total_downtime_hours_90d": round(float(top["total_downtime_hours"]), 1),
            "most_common_component": top["most_common_component"],
            "recommended_action": "Schedule inspection" if needs_inspection else "Continue routine monitoring",
            "top5_ranking": ranking.to_dict(orient="records"),
        }
        explanation = (
            f"{top['turbine_id']} has the highest risk score ({top['risk_score']:.1f}) over the last "
            f"90 days: {int(top['n_failures'])} failures, {top['total_downtime_hours']:.0f}h downtime, "
            f"most affected component '{top['most_common_component']}'. "
            f"Recommendation: {output['recommended_action']}. "
            f"[Rule-based risk score from real alarm/maintenance data, not a trained failure model.]"
        )
        return AgentResult(agent_name=self.name, output=output, explanation=explanation)
