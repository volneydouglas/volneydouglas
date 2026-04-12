"""
ExperimentRunner — orchestrates all experimental conditions and sessions.

Runs the full 2×2×3 factorial design:
  - 12 conditions × N sessions per condition
  - Saves raw data after each session for crash resilience
  - Supports resumption from partial runs
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import Settings
from games import GAME_REGISTRY
from frames import FrameType, PartnerType, IdentityFrame
from experiment.conditions import ExperimentalCondition, generate_all_conditions
from experiment.session import GameSession, SessionResult


class ExperimentRunner:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.conditions = generate_all_conditions()
        self.results: list[SessionResult] = []
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.settings.experiment.output_dir) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _completed_sessions(self, condition: ExperimentalCondition) -> int:
        """Count how many sessions have been completed for a condition."""
        pattern = f"{condition.label}_session_*.json"
        return len(list(self.output_dir.glob(pattern)))

    def _save_session(self, result: SessionResult) -> None:
        """Save a single session result to disk."""
        filename = f"{result.condition_label}_session_{result.session_id}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def _save_manifest(self) -> None:
        """Save a manifest of the experiment run."""
        manifest = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "settings": {
                "model": self.settings.model.model_id,
                "temperature": self.settings.model.temperature,
                "rounds_per_session": self.settings.experiment.rounds_per_session,
                "sessions_per_condition": self.settings.experiment.sessions_per_condition,
            },
            "conditions": [c.label for c in self.conditions],
            "total_conditions": len(self.conditions),
            "total_sessions_planned": (
                len(self.conditions) * self.settings.experiment.sessions_per_condition
            ),
            "total_sessions_completed": len(self.results),
        }
        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def run_condition(
        self, condition: ExperimentalCondition, progress_callback=None
    ) -> list[SessionResult]:
        """Run all sessions for a single experimental condition."""
        game_cls = GAME_REGISTRY[condition.game_type]
        game = game_cls()
        frame = condition.identity_frame
        sessions_needed = self.settings.experiment.sessions_per_condition
        completed = self._completed_sessions(condition)
        condition_results = []

        for i in range(completed, sessions_needed):
            session_id = str(uuid.uuid4())[:8]
            session = GameSession(
                session_id=session_id,
                game=game,
                frame=frame,
                settings=self.settings,
            )
            result = session.run(self.settings.experiment.rounds_per_session)
            self._save_session(result)
            self.results.append(result)
            condition_results.append(result)

            if progress_callback:
                progress_callback(condition, i + 1, sessions_needed, result)

        return condition_results

    def run_all(self, progress_callback=None) -> list[SessionResult]:
        """Run the complete experiment across all 12 conditions."""
        self.settings.validate()
        self._save_manifest()

        total = len(self.conditions)
        for idx, condition in enumerate(self.conditions, 1):
            print(f"\n[{idx}/{total}] Running condition: {condition.label}")
            self.run_condition(condition, progress_callback)

        self._save_manifest()  # Update with final counts
        self._save_summary()
        print(f"\nExperiment complete. Results saved to {self.output_dir}")
        return self.results

    def _save_summary(self) -> None:
        """Save an aggregate summary of all results."""
        summary = {}
        for result in self.results:
            label = result.condition_label
            if label not in summary:
                summary[label] = {
                    "frame_type": result.frame_type,
                    "partner_type": result.partner_type,
                    "game_type": result.game_type,
                    "sessions": 0,
                    "cooperation_rates": [],
                    "total_payoffs": [],
                }
            summary[label]["sessions"] += 1
            summary[label]["cooperation_rates"].append(result.cooperation_rate)
            summary[label]["total_payoffs"].append(result.total_payoff)

        # Compute means
        for label, data in summary.items():
            rates = data["cooperation_rates"]
            payoffs = data["total_payoffs"]
            data["mean_cooperation_rate"] = sum(rates) / len(rates) if rates else 0
            data["mean_total_payoff"] = sum(payoffs) / len(payoffs) if payoffs else 0

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def run_single(
        self,
        frame_type: str,
        partner_type: str,
        game_type: str,
        num_sessions: int = 1,
    ) -> list[SessionResult]:
        """Run a specific condition for quick testing."""
        self.settings.validate()
        condition = ExperimentalCondition(
            frame_type=FrameType(frame_type),
            partner_type=PartnerType(partner_type),
            game_type=game_type,
        )
        game_cls = GAME_REGISTRY[condition.game_type]
        game = game_cls()
        frame = condition.identity_frame

        results = []
        for i in range(num_sessions):
            session_id = str(uuid.uuid4())[:8]
            session = GameSession(
                session_id=session_id,
                game=game,
                frame=frame,
                settings=self.settings,
            )
            result = session.run(self.settings.experiment.rounds_per_session)
            self._save_session(result)
            results.append(result)
            print(
                f"  Session {i+1}/{num_sessions}: "
                f"cooperation={result.cooperation_rate:.0%}, "
                f"payoff={result.total_payoff}"
            )
        return results
