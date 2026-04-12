"""
Experimental conditions — 2 × 2 × 3 factorial design.

IV1: Identity Frame (extended_phenotype | separate_entity)
IV2: Partner Knowledge (human | ai)
IV3: Game Type (prisoners_dilemma | stag_hunt | battle_of_sexes)

Total: 2 × 2 × 3 = 12 unique conditions
"""

from dataclasses import dataclass
from itertools import product

from frames import FrameType, PartnerType, IdentityFrame
from games import GameType


@dataclass(frozen=True)
class ExperimentalCondition:
    frame_type: FrameType
    partner_type: PartnerType
    game_type: GameType

    @property
    def label(self) -> str:
        return f"{self.frame_type.value}__{self.partner_type.value}__{self.game_type.value}"

    @property
    def identity_frame(self) -> IdentityFrame:
        return IdentityFrame(self.frame_type, self.partner_type)


def generate_all_conditions() -> list[ExperimentalCondition]:
    """Generate all 12 conditions from the factorial design."""
    conditions = []
    for frame, partner, game in product(FrameType, PartnerType, GameType):
        conditions.append(ExperimentalCondition(frame, partner, game))
    return conditions
