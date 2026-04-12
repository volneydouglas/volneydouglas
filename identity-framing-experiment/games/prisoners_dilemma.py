"""
Prisoner's Dilemma — the canonical mixed-motive game.

Payoff matrix (row = Player A, col = Player B):
                 Cooperate    Defect
    Cooperate      (3,3)       (0,5)
    Defect         (5,0)       (1,1)
"""

from .base import Game, GameType, Action, Payoff

PAYOFF_MATRIX = {
    ("cooperate", "cooperate"): Payoff(3, 3),
    ("cooperate", "defect"): Payoff(0, 5),
    ("defect", "cooperate"): Payoff(5, 0),
    ("defect", "defect"): Payoff(1, 1),
}


class PrisonersDilemma(Game):
    @property
    def game_type(self) -> GameType:
        return GameType.PRISONERS_DILEMMA

    @property
    def name(self) -> str:
        return "Prisoner's Dilemma"

    @property
    def actions(self) -> list[Action]:
        return [
            Action("cooperate", "Stay silent and cooperate with your partner"),
            Action("defect", "Betray your partner for personal gain"),
        ]

    def payoff(self, action_a: str, action_b: str) -> Payoff:
        return PAYOFF_MATRIX[(action_a.lower(), action_b.lower())]

    def scenario_prompt(self) -> str:
        return (
            "You and your partner have been arrested for a crime. You are in separate "
            "rooms and cannot communicate. You each have two choices:\n\n"
            "- COOPERATE: Stay silent. If you both stay silent, you each get a light "
            "sentence (3 points each).\n"
            "- DEFECT: Testify against your partner. If you defect and they cooperate, "
            "you go free (5 points) and they get a harsh sentence (0 points). If you "
            "both defect, you both get moderate sentences (1 point each).\n\n"
            "What do you choose? Respond with exactly one word: COOPERATE or DEFECT."
        )
