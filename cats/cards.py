from dataclasses import dataclass
import random


CAT_TYPES = [
    "alive",
    "dead",
    "empty",
    "heisenberg",
]


@dataclass(frozen=True)
class Card:
    cat_type: str

    def __post_init__(self):
        if self.cat_type not in CAT_TYPES:
            raise ValueError(f"Unknown cat type: {self.cat_type}")

    def supports_claim(self, claim_type: str, heisenbergs_count: bool = True) -> bool:
        if self.cat_type == claim_type:
            return True

        if self.cat_type == "heisenberg" and heisenbergs_count:
            return True

        return False

    def __repr__(self):
        symbols = {
            "alive": "A",
            "dead": "D",
            "empty": "E",
            "heisenberg": "H",
        }
        return symbols[self.cat_type]


class Deck:
    def __init__(self, cards=None):
        if cards is None:
            self.cards = self._create_default_deck()
        else:
            self.cards = list(cards)

    def _create_default_deck(self):
        cards = []

        cards += [Card("alive") for _ in range(20)]
        cards += [Card("dead") for _ in range(20)]
        cards += [Card("empty") for _ in range(8)]
        cards += [Card("heisenberg") for _ in range(4)]

        return cards

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, amount=1):
        drawn = []

        for _ in range(amount):
            if not self.cards:
                break

            drawn.append(self.cards.pop())

        return drawn

    def add_cards(self, cards):
        self.cards.extend(cards)

    def count(self, cat_type):
        return sum(1 for card in self.cards if card.cat_type == cat_type)

    def __len__(self):
        return len(self.cards)

    def __repr__(self):
        return f"Deck({len(self.cards)} cards)"