"""
Base classes for game-theoretic games used in the experiment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class GameType(StrEnum):
    PRISONERS_DILEMMA = "prisoners_dilemma"
    STAG_HUNT = "stag_hunt"
    BATTLE_OF_SEXES = "battle_of_sexes"


@dataclass(frozen=True)
class Action:
    name: str
    description: str


@dataclass(frozen=True)
class Payoff:
    player_a: float
    player_b: float


class Game(ABC):
    """Base class for all two-player symmetric/asymmetric games."""

    @property
    @abstractmethod
    def game_type(self) -> GameType:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def actions(self) -> list[Action]:
        ...

    @abstractmethod
    def payoff(self, action_a: str, action_b: str) -> Payoff:
        """Return payoffs for both players given their chosen actions."""
        ...

    @abstractmethod
    def scenario_prompt(self) -> str:
        """Return a natural-language description of the game for the LLM."""
        ...

    @property
    def cooperative_action(self) -> str:
        """The action considered 'cooperative' for analysis purposes."""
        return self.actions[0].name

    def is_cooperative(self, action: str) -> bool:
        return action == self.cooperative_action

    @property
    def action_names(self) -> list[str]:
        return [a.name for a in self.actions]
