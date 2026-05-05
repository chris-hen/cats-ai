from dataclasses import dataclass, field
import random

from cats.cards import Deck, Card
from cats.physicats import PhysicatDeck, Physicat

MIN_PLAYERS = 2
MAX_PLAYERS = 6

CLAIM_TYPES = ["alive", "dead", "empty"]


@dataclass
class Claim:
    amount: int
    cat_type: str
    player_id: int

    def __post_init__(self):
        if self.cat_type not in CLAIM_TYPES:
            raise ValueError(f"Invalid claim type: {self.cat_type}")

    def __repr__(self):
        symbols = {
            "alive": "A",
            "dead": "D",
            "empty": "E",
        }
        return f"{self.amount}{symbols[self.cat_type]} by P{self.player_id}"


@dataclass
class Player:
    player_id: int
    hand: list[Card] = field(default_factory=list)
    proofs: list[Card] = field(default_factory=list)
    active: bool = True
    exists: bool = True
    lives: int = 1
    physicat: Physicat | None = None
    physicat_used: bool = False
    physicat_proofs: dict[str, int] = field(default_factory=lambda: {
        "alive": 0,
        "dead": 0,
        "empty": 0,
    })
    known_discard: list[Card] = field(default_factory=list)
    saw_full_discard: bool = False
    round_physicats: list[Physicat] = field(default_factory=list)

    def __repr__(self):
        if not self.exists:
            return f"P{self.player_id}(missing)"

        status = "active" if self.active else "out"
        return (
            f"P{self.player_id}("
            f"{status}, lives={self.lives}, "
            f"physicat={self.physicat}, used={self.physicat_used}, "
            f"hand={self.hand}, proofs={self.proofs})"
        )


@dataclass
class GameState:
    num_players: int
    round_number: int = 1
    cards_per_player: int = 6
    current_player: int = 0
    starting_player: int = 0
    current_claim: Claim | None = None
    heisenbergs_count: bool = True
    deck: Deck = field(default_factory=Deck)
    discard_pile: list[Card] = field(default_factory=list)
    face_up_physicats: list[Physicat] = field(default_factory=list)
    players: list[Player] = field(default_factory=list)

    def __post_init__(self):
        if self.num_players < MIN_PLAYERS or self.num_players > MAX_PLAYERS:
            raise ValueError(f"num_players must be between {MIN_PLAYERS} and {MAX_PLAYERS}")

        if not self.players:
            starting_lives = self.starting_lives_for_player_count(self.num_players)

            self.players = []

            for i in range(MAX_PLAYERS):
                exists = i < self.num_players

                self.players.append(
                    Player(
                        player_id=i,
                        exists=exists,
                        active=exists,
                        lives=starting_lives if exists else 0,
                    )
                )

    @staticmethod
    def starting_lives_for_player_count(num_players):
        if num_players == 2:
            return 3
        if num_players == 3:
            return 2
        return 1

    def existing_players(self):
        return [p for p in self.players if p.exists]

    def active_players(self):
        return [p for p in self.players if p.exists and p.active]

    def active_player_ids(self):
        return [p.player_id for p in self.active_players()]

    def is_game_over(self):
        return len(self.active_players()) <= 1

    def winner(self):
        active = self.active_players()
        if len(active) == 1:
            return active[0].player_id
        return None

    def setup_round(self):
        if self.round_number == 1:
            self.setup_physicats()

        for player in self.players:
            if player.exists and player.round_physicats:
                self.face_up_physicats.extend(player.round_physicats)
                player.round_physicats = []

        self.deck = Deck()
        self.deck.shuffle()
        self.discard_pile = []
        self.current_claim = None
        self.heisenbergs_count = True

        for player in self.players:
            player.hand = []
            player.proofs = []
            player.physicat_proofs = {
                "alive": 0,
                "dead": 0,
                "empty": 0,
            }
            player.known_discard = []
            player.saw_full_discard = False
            player.round_physicats = []

            if player.exists and player.active:
                player.hand = self.deck.draw(self.cards_per_player)

        if self.round_number == 1:
            self.starting_player = random.choice(self.active_player_ids())

        self.current_player = self.starting_player

    def next_active_player(self, player_id=None):
        if player_id is None:
            player_id = self.current_player

        active_ids = self.active_player_ids()

        if not active_ids:
            return None

        next_id = (player_id + 1) % MAX_PLAYERS

        while next_id not in active_ids:
            next_id = (next_id + 1) % MAX_PLAYERS

        return next_id

    def advance_turn(self):
        self.current_player = self.next_active_player(self.current_player)

    def total_cards_in_play(self):
        total = 0

        for player in self.active_players():
            total += len(player.hand)
            total += len(player.proofs)

        return total

    def reveal_count_for_claim(self, claim: Claim, heisenbergs_count: bool = True):
        count = 0

        for player in self.active_players():
            all_cards = player.hand + player.proofs

            for card in all_cards:
                if card.supports_claim(claim.cat_type, heisenbergs_count):
                    count += 1

            count += player.physicat_proofs.get(claim.cat_type, 0)

        return count

    def eliminate_or_damage_player(self, player_id):
        player = self.players[player_id]

        if not player.exists or not player.active:
            return

        player.lives -= 1

        if player.lives <= 0:
            # Unbenutzte Physicat wird beim Ausscheiden offen gelegt
            if player.physicat is not None and not player.physicat_used:
                self.face_up_physicats.append(player.physicat)
                player.physicat_used = True

            # Falls eine round_physicat vor dem Spieler liegt, wandert sie ebenfalls in die offene Reihe
            if player.round_physicats:
                self.face_up_physicats.extend(player.round_physicats)
                player.round_physicats = []

            player.active = False
            player.hand = []
            player.proofs = []
            player.physicat_proofs = {
                "alive": 0,
                "dead": 0,
                "empty": 0,
            }

    def start_next_round(self, starting_player):
        self.round_number += 1
        self.cards_per_player = max(1, self.cards_per_player - 1)
        self.starting_player = starting_player
        self.setup_round()

    def setup_physicats(self):
        deck = PhysicatDeck()
        deck.shuffle()

        for player in self.players:
            player.physicat = None
            player.physicat_used = False

        for player in self.existing_players():
            player.physicat = deck.draw(1)[0]

        self.face_up_physicats = deck.draw(1)