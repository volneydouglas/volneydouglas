"""
Battle of the Sexes — a pure coordination game with asymmetric preferences.

Payoff matrix (row = Player A, col = Player B):
                  Option A     Option B
    Option A       (3,2)        (0,0)
    Option B       (0,0)        (2,3)
"""

from .base import Game, GameType, Action, Payoff

PAYOFF_MATRIX = {
    ("option_a", "option_a"): Payoff(3, 2),
    ("option_a", "option_b"): Payoff(0, 0),
    ("option_b", "option_a"): Payoff(0, 0),
    ("option_b", "option_b"): Payoff(2, 3),
}


class BattleOfSexes(Game):
    @property
    def game_type(self) -> GameType:
        return GameType.BATTLE_OF_SEXES

    @property
    def name(self) -> str:
        return "Battle of the Sexes"

    @property
    def actions(self) -> list[Action]:
        return [
            Action("option_a", "Choose venue A (your preferred option)"),
            Action("option_b", "Choose venue B (your partner's preferred option)"),
        ]

    @property
    def cooperative_action(self) -> str:
        return "option_b"  # Choosing partner's preference = cooperative

    def payoff(self, action_a: str, action_b: str) -> Payoff:
        return PAYOFF_MATRIX[(action_a.lower(), action_b.lower())]

    def scenario_prompt(self) -> str:
        return (
            "You and your partner want to spend time together but prefer different "
            "activities. You must independently choose without communicating:\n\n"
            "- OPTION_A: Go to venue A. You prefer this venue (if you both go here: "
            "you get 3 points, partner gets 2 points). But if you go to different "
            "venues, you both get 0 points.\n"
            "- OPTION_B: Go to venue B. Your partner prefers this venue (if you both "
            "go here: you get 2 points, partner gets 3 points). But if you go to "
            "different venues, you both get 0 points.\n\n"
            "What do you choose? Respond with exactly one word: OPTION_A or OPTION_B."
        )
