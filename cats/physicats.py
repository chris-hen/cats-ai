from dataclasses import dataclass
import random


PHYSICAT_TYPES = [
    "cecilia",   # +2 alive
    "maria",     # +1 empty
    "marie",     # remove alive proofs
    "michael",   # remove dead proofs
    "richard",   # heisenbergs don't count
    "sally",     # look at discard
    "newton",    # draw 2
    "albert",    # swap entire hand
    "stephen",   # skip turn
    "neil",      # use face-up physicat ability
]


@dataclass(frozen=True)
class Physicat:
    physicat_type: str

    def __post_init__(self):
        if self.physicat_type not in PHYSICAT_TYPES:
            raise ValueError(f"Unknown Physicat: {self.physicat_type}")

    def __repr__(self):
        names = {
            "cecilia": "Cecilia(+2A)",
            "maria": "Maria(+1E)",
            "marie": "Marie(remove A proofs)",
            "michael": "Michael(remove D proofs)",
            "richard": "Richard(no H)",
            "sally": "Sally(look discard)",
            "newton": "Newton(draw 2)",
            "albert": "Albert(swap hand)",
            "stephen": "Stephen(skip)",
            "neil": "Neil(copy)",
        }
        return names[self.physicat_type]


class PhysicatDeck:
    def __init__(self):
        self.cards = [Physicat(t) for t in PHYSICAT_TYPES]

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, amount=1):
        drawn = []

        for _ in range(amount):
            if not self.cards:
                break
            drawn.append(self.cards.pop())

        return drawn