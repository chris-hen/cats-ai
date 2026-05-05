import math
import random

from cats.actions import MakeClaimAction, DoubtAction, UsePhysicatAction
from cats.rules import legal_actions
from cats.cards import CAT_TYPES


FULL_DECK_COUNTS = {
    "alive": 20,
    "dead": 20,
    "empty": 8,
    "heisenberg": 4,
}


def count_card_types(cards):
    counts = {t: 0 for t in CAT_TYPES}
    for card in cards:
        counts[card.cat_type] += 1
    return counts


def subtract_counts(a, b):
    result = dict(a)
    for k, v in b.items():
        result[k] -= v
    return result


def hypergeom_at_least(population_size, success_count, draws, needed_successes):
    if needed_successes <= 0:
        return 1.0
    if draws <= 0:
        return 0.0
    if success_count <= 0:
        return 0.0

    max_successes = min(success_count, draws)
    if needed_successes > max_successes:
        return 0.0

    total = math.comb(population_size, draws)
    prob = 0.0

    for k in range(needed_successes, max_successes + 1):
        if draws - k > population_size - success_count:
            continue

        prob += (
            math.comb(success_count, k)
            * math.comb(population_size - success_count, draws - k)
            / total
        )

    return prob


class BayesianBaseBot:
    """
    Gemeinsame Bayesian-Basis.

    Enthält:
    - Claim-Wahrscheinlichkeiten
    - Hidden-card Modell
    - Action-Auswahl
    - Cache
    """

    def __init__(self, random_noise=0.01):
        self.random_noise = random_noise

    def choose_action(self, state):
        actions = legal_actions(state)

        if not actions:
            raise RuntimeError("No legal actions available")

        player_id = state.current_player
        self._prob_cache = {}

        best_by_key = {}

        for action in actions:
            if isinstance(action, MakeClaimAction):
                key = ("claim", action.amount, action.cat_type)
            elif isinstance(action, DoubtAction):
                key = ("doubt",)
            elif isinstance(action, UsePhysicatAction):
                key = ("physicat", action.target_index)
            else:
                key = ("other", repr(action))

            score = self.score_action(state, player_id, action)

            if key not in best_by_key or score > best_by_key[key][0]:
                best_by_key[key] = (score, action)

        scored = list(best_by_key.values())
        scored.sort(key=lambda x: x[0], reverse=True)

        self._prob_cache = {}

        return scored[0][1]

    def known_cards_for_player(self, state, player_id):
        player = state.players[player_id]

        known = []
        known.extend(player.hand)

        for p in state.active_players():
            known.extend(p.proofs)

        known.extend(player.known_discard)

        return known

    def known_support_in_play(self, state, player_id, cat_type):
        count = 0
        player = state.players[player_id]

        for card in player.hand:
            if card.supports_claim(cat_type, state.heisenbergs_count):
                count += 1

        for p in state.active_players():
            for card in p.proofs:
                if card.supports_claim(cat_type, state.heisenbergs_count):
                    count += 1

            count += p.physicat_proofs.get(cat_type, 0)

        return count

    def unknown_pool_counts(self, state, player_id):
        known_cards = self.known_cards_for_player(state, player_id)
        known_counts = count_card_types(known_cards)
        return subtract_counts(FULL_DECK_COUNTS, known_counts)

    def support_count_in_pool(self, pool_counts, cat_type, heisenbergs_count=True):
        support = pool_counts[cat_type]

        if heisenbergs_count:
            support += pool_counts["heisenberg"]

        return support

    def claim_truth_probability(self, state, player_id, amount, cat_type):
        cache_key = (player_id, amount, cat_type, state.heisenbergs_count)

        if hasattr(self, "_prob_cache") and cache_key in self._prob_cache:
            return self._prob_cache[cache_key]

        known_support = self.known_support_in_play(state, player_id, cat_type)
        needed_from_hidden = amount - known_support

        if needed_from_hidden <= 0:
            result = 1.0
        else:
            hidden_cards_in_play = 0

            for p in state.active_players():
                if p.player_id != player_id:
                    hidden_cards_in_play += len(p.hand)

            pool_counts = self.unknown_pool_counts(state, player_id)
            population_size = sum(pool_counts.values())
            success_count = self.support_count_in_pool(
                pool_counts,
                cat_type,
                state.heisenbergs_count,
            )

            result = hypergeom_at_least(
                population_size=population_size,
                success_count=success_count,
                draws=hidden_cards_in_play,
                needed_successes=needed_from_hidden,
            )

        if hasattr(self, "_prob_cache"):
            self._prob_cache[cache_key] = result

        return result

    def score_action(self, state, player_id, action):
        noise = random.uniform(-self.random_noise, self.random_noise)

        if isinstance(action, DoubtAction):
            return self.score_doubt(state, player_id) + noise

        if isinstance(action, MakeClaimAction):
            return self.score_claim(state, player_id, action) + noise

        if isinstance(action, UsePhysicatAction):
            return self.score_physicat(state, player_id, action) + noise

        return -999.0

    def score_claim(self, state, player_id, action):
        raise NotImplementedError

    def score_doubt(self, state, player_id):
        raise NotImplementedError

    def score_physicat(self, state, player_id, action):
        raise NotImplementedError