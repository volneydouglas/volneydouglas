"""
GameSession — runs a single multi-round session between an LLM and a simulated partner.

Each session:
  1. Sets the system prompt based on the identity frame condition
  2. Plays N rounds of the specified game
  3. Records all decisions, payoffs, and raw LLM responses
"""

import json
import re
import time
from dataclasses import dataclass, field

from anthropic import Anthropic

from config import Settings
from frames import IdentityFrame, build_system_prompt
from frames.identity_frames import build_round_prompt
from games.base import Game


@dataclass
class RoundResult:
    round_num: int
    player_action: str
    partner_action: str
    player_payoff: float
    partner_payoff: float
    raw_response: str
    response_time_ms: float
    is_cooperative: bool


@dataclass
class SessionResult:
    session_id: str
    condition_label: str
    frame_type: str
    partner_type: str
    game_type: str
    rounds: list[RoundResult] = field(default_factory=list)

    @property
    def cooperation_rate(self) -> float:
        if not self.rounds:
            return 0.0
        return sum(1 for r in self.rounds if r.is_cooperative) / len(self.rounds)

    @property
    def total_payoff(self) -> float:
        return sum(r.player_payoff for r in self.rounds)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "condition_label": self.condition_label,
            "frame_type": self.frame_type,
            "partner_type": self.partner_type,
            "game_type": self.game_type,
            "cooperation_rate": self.cooperation_rate,
            "total_payoff": self.total_payoff,
            "rounds": [
                {
                    "round": r.round_num,
                    "player_action": r.player_action,
                    "partner_action": r.partner_action,
                    "player_payoff": r.player_payoff,
                    "partner_payoff": r.partner_payoff,
                    "raw_response": r.raw_response,
                    "response_time_ms": r.response_time_ms,
                    "is_cooperative": r.is_cooperative,
                }
                for r in self.rounds
            ],
        }


class GameSession:
    """Manages a single experimental session (one LLM playing N rounds)."""

    def __init__(
        self,
        session_id: str,
        game: Game,
        frame: IdentityFrame,
        settings: Settings,
        partner_strategy: str = "tit_for_tat",
    ):
        self.session_id = session_id
        self.game = game
        self.frame = frame
        self.settings = settings
        self.partner_strategy = partner_strategy
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.system_prompt = build_system_prompt(frame)
        self.history: list[dict] = []

    def _get_partner_action(self, round_num: int) -> str:
        """Simulate a partner using a fixed strategy."""
        actions = self.game.action_names
        if self.partner_strategy == "always_cooperate":
            return actions[0]
        elif self.partner_strategy == "always_defect":
            return actions[-1]
        elif self.partner_strategy == "tit_for_tat":
            if round_num == 1 or not self.history:
                return actions[0]  # Start cooperative
            return self.history[-1]["your_action"]  # Mirror last player action
        elif self.partner_strategy == "random":
            import random
            return random.choice(actions)
        return actions[0]

    def _parse_action(self, raw_response: str) -> str:
        """Extract a valid action from the LLM's response."""
        cleaned = raw_response.strip().upper()
        for action in self.game.action_names:
            if action.upper() in cleaned:
                return action
        # Fallback: try partial matching
        for action in self.game.action_names:
            if action.upper()[:4] in cleaned:
                return action
        return self.game.action_names[0]  # Default to first action

    def run(self, num_rounds: int) -> SessionResult:
        """Execute the full session and return results."""
        result = SessionResult(
            session_id=self.session_id,
            condition_label=self.frame.condition_label,
            frame_type=self.frame.frame_type.value,
            partner_type=self.frame.partner_type.value,
            game_type=self.game.game_type.value,
        )

        messages = []

        for round_num in range(1, num_rounds + 1):
            user_prompt = build_round_prompt(
                self.game.scenario_prompt(), round_num, self.history
            )
            messages.append({"role": "user", "content": user_prompt})

            start_time = time.time()
            response = self.client.messages.create(
                model=self.settings.model.model_id,
                max_tokens=self.settings.model.max_tokens,
                temperature=self.settings.model.temperature,
                system=self.system_prompt,
                messages=messages,
            )
            elapsed_ms = (time.time() - start_time) * 1000

            raw_text = response.content[0].text
            player_action = self._parse_action(raw_text)
            partner_action = self._get_partner_action(round_num)
            payoffs = self.game.payoff(player_action, partner_action)

            round_result = RoundResult(
                round_num=round_num,
                player_action=player_action,
                partner_action=partner_action,
                player_payoff=payoffs.player_a,
                partner_payoff=payoffs.player_b,
                raw_response=raw_text,
                response_time_ms=elapsed_ms,
                is_cooperative=self.game.is_cooperative(player_action),
            )
            result.rounds.append(round_result)

            # Update history for future rounds
            self.history.append({
                "round": round_num,
                "your_action": player_action,
                "partner_action": partner_action,
                "your_payoff": payoffs.player_a,
            })

            # Add assistant response to messages for conversational continuity
            messages.append({"role": "assistant", "content": raw_text})

        return result
