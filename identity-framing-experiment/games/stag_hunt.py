"""
Stag Hunt — a coordination game with payoff-dominant and risk-dominant equilibria.

Payoff matrix (row = Player A, col = Player B):
                  Stag        Hare
    Stag          (4,4)       (0,3)
    Hare          (3,0)       (3,3)
"""

from .base import Game, GameType, Action, Payoff

PAYOFF_MATRIX = {
    ("stag", "stag"): Payoff(4, 4),
    ("stag", "hare"): Payoff(0, 3),
    ("hare", "stag"): Payoff(3, 0),
    ("hare", "hare"): Payoff(3, 3),
}


class StagHunt(Game):
    @property
    def game_type(self) -> GameType:
        return GameType.STAG_HUNT

    @property
    def name(self) -> str:
        return "Stag Hunt"

    @property
    def actions(self) -> list[Action]:
        return [
            Action("stag", "Hunt the stag together for a larger reward"),
            Action("hare", "Hunt a hare alone for a safe but smaller reward"),
        ]

    def payoff(self, action_a: str, action_b: str) -> Payoff:
        return PAYOFF_MATRIX[(action_a.lower(), action_b.lower())]

    def scenario_prompt(self) -> str:
        return (
            "You and your partner are hunters. You must independently decide what to "
            "hunt:\n\n"
            "- STAG: Hunt a stag. This requires coordination — if you both choose stag, "
            "you share a large meal (4 points each). But if you choose stag and your "
            "partner chooses hare, you get nothing (0 points).\n"
            "- HARE: Hunt a hare alone. You are guaranteed a small meal (3 points) "
            "regardless of what your partner does.\n\n"
            "What do you choose? Respond with exactly one word: STAG or HARE."
        )
