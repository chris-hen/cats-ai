import math
import random

from cats.actions import MakeClaimAction, DoubtAction, UsePhysicatAction
from cats.rules import legal_actions, claim_rank
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


class BayesianHeuristicBot:
    """
    Erste starke Baseline.

    Idee:
    - Schätzt P(aktueller Claim ist wahr)
    - Doubtet, wenn Claim wahrscheinlich falsch ist
    - Sonst wählt er einen legalen Claim mit guter Balance aus:
      Wahrscheinlichkeit, Rangfortschritt, Proof-Kosten, Swap-Kosten
    """

    def __init__(
        self,
        doubt_threshold=0.42,
        claim_confidence_weight=3.0,
        claim_rank_weight=0.025,
        proof_penalty=0.08,
        swap_penalty=0.04,
        physicat_bonus=0.15,
        random_noise=0.01,
    ):
        self.doubt_threshold = doubt_threshold
        self.claim_confidence_weight = claim_confidence_weight
        self.claim_rank_weight = claim_rank_weight
        self.proof_penalty = proof_penalty
        self.swap_penalty = swap_penalty
        self.physicat_bonus = physicat_bonus
        self.random_noise = random_noise

    def choose_action(self, state):
        actions = legal_actions(state)

        if not actions:
            raise RuntimeError("No legal actions available")

        player_id = state.current_player

        # Cache für Claim-Wahrscheinlichkeiten pro Zug
        self._prob_cache = {}

        # Begrenze kombinatorische Explosion:
        # Pro Claim behalten wir nur die beste Proof/Swap-Variante.
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

        # Eigene Hand kennt man.
        known.extend(player.hand)

        # Alle offenen Proofs sind bekannt.
        for p in state.active_players():
            known.extend(p.proofs)

        # Eigener bekannter Discard, z. B. durch eigene Swaps oder Sally.
        known.extend(player.known_discard)

        return known

    def known_support_in_play(self, state, player_id, cat_type):
        count = 0

        # Eigene Hand zählt sicher.
        player = state.players[player_id]
        for card in player.hand:
            if card.supports_claim(cat_type, state.heisenbergs_count):
                count += 1

        # Offene Proofs zählen sicher.
        for p in state.active_players():
            for card in p.proofs:
                if card.supports_claim(cat_type, state.heisenbergs_count):
                    count += 1

            # Physicat-Proofs sind keine Karten, zählen aber für den Claim.
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

    def score_doubt(self, state, player_id):
        claim = state.current_claim

        if claim is None:
            return -999.0

        p_true = self.claim_truth_probability(
            state,
            player_id,
            claim.amount,
            claim.cat_type,
        )

        p_false = 1.0 - p_true

        # Je klarer p_false über der Schwelle liegt, desto besser Doubt.
        return (p_false - self.doubt_threshold) * 4.0

    def score_claim(self, state, player_id, action):
        p_true = self.claim_truth_probability(
            state,
            player_id,
            action.amount,
            action.cat_type,
        )

        rank = claim_rank(action.amount, action.cat_type)

        score = 0.0

        # Claims sollen wahrscheinlich genug sein.
        score += self.claim_confidence_weight * p_true

        # Aber nicht zu passiv: höhere Claims bekommen leichten Bonus.
        score += self.claim_rank_weight * rank

        # Proofs geben Information preis.
        score -= self.proof_penalty * len(action.proof_indices)

        # Swaps sind nützlich, aber unsicher.
        score -= self.swap_penalty * len(action.swap_indices)

        # Sehr unwahrscheinliche Claims stark bestrafen.
        if p_true < 0.25:
            score -= 2.0

        return score

    def score_physicat(self, state, player_id, action):
        player = state.players[player_id]
        physicat = player.physicat

        if physicat is None:
            return -999.0

        t = physicat.physicat_type

        # Einfache erste Heuristik.
        bonuses = {
            "newton": 0.45,
            "albert": 0.25,
            "sally": 0.20,
            "cecilia": 0.30,
            "maria": 0.25,
            "richard": 0.20,
            "marie": 0.15,
            "michael": 0.15,
            "stephen": 0.10,
            "neil": 0.25,
        }

        return self.physicat_bonus + bonuses.get(t, 0.0)