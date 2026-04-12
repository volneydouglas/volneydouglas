"""
Compute descriptive metrics from experiment results.
"""

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ConditionMetrics:
    condition_label: str
    frame_type: str
    partner_type: str
    game_type: str
    n_sessions: int
    cooperation_rates: list[float]
    total_payoffs: list[float]
    round_by_round_cooperation: list[float]  # Cooperation rate per round position

    @property
    def mean_cooperation(self) -> float:
        return sum(self.cooperation_rates) / len(self.cooperation_rates)

    @property
    def std_cooperation(self) -> float:
        mean = self.mean_cooperation
        variance = sum((x - mean) ** 2 for x in self.cooperation_rates) / len(
            self.cooperation_rates
        )
        return variance**0.5

    @property
    def mean_payoff(self) -> float:
        return sum(self.total_payoffs) / len(self.total_payoffs)


def load_session_files(results_dir: str | Path) -> list[dict]:
    """Load all session JSON files from a results directory."""
    results_path = Path(results_dir)
    sessions = []
    for filepath in sorted(results_path.glob("*_session_*.json")):
        with open(filepath) as f:
            sessions.append(json.load(f))
    return sessions


def compute_condition_metrics(results_dir: str | Path) -> dict[str, ConditionMetrics]:
    """Aggregate session data into per-condition metrics."""
    sessions = load_session_files(results_dir)
    grouped: dict[str, list[dict]] = {}
    for s in sessions:
        label = s["condition_label"]
        grouped.setdefault(label, []).append(s)

    metrics = {}
    for label, group in grouped.items():
        coop_rates = [s["cooperation_rate"] for s in group]
        payoffs = [s["total_payoff"] for s in group]

        # Round-by-round cooperation
        max_rounds = max(len(s["rounds"]) for s in group)
        round_coop = []
        for r in range(max_rounds):
            cooperated = sum(
                1
                for s in group
                if r < len(s["rounds"]) and s["rounds"][r]["is_cooperative"]
            )
            total = sum(1 for s in group if r < len(s["rounds"]))
            round_coop.append(cooperated / total if total > 0 else 0.0)

        sample = group[0]
        metrics[label] = ConditionMetrics(
            condition_label=label,
            frame_type=sample["frame_type"],
            partner_type=sample["partner_type"],
            game_type=sample["game_type"],
            n_sessions=len(group),
            cooperation_rates=coop_rates,
            total_payoffs=payoffs,
            round_by_round_cooperation=round_coop,
        )

    return metrics
