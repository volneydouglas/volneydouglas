from .base import Game, GameType, Action, Payoff
from .prisoners_dilemma import PrisonersDilemma
from .stag_hunt import StagHunt
from .battle_of_sexes import BattleOfSexes

GAME_REGISTRY: dict[str, type[Game]] = {
    GameType.PRISONERS_DILEMMA: PrisonersDilemma,
    GameType.STAG_HUNT: StagHunt,
    GameType.BATTLE_OF_SEXES: BattleOfSexes,
}
