from bots.bayesian_base import BayesianBaseBot
from cats.rules import claim_rank


class BayesianV3Bot(BayesianBaseBot):
    """
    v3:
    - eigene Scoringlogik auf gemeinsamer Base
    - Handqualität
    - dynamische Proof-/Swap-Bewertung
    - Endgame-Anpassung
    - bessere Physicat-Bewertung
    """

    def __init__(self, doubt_threshold=0.34, random_noise=0.01):
        super().__init__(random_noise=random_noise)
        self.doubt_threshold = doubt_threshold

    def count_support_in_cards(self, cards, cat_type, heisenbergs_count=True):
        return sum(
            1 for card in cards
            if card.supports_claim(cat_type, heisenbergs_count)
        )

    def hand_quality(self, state, player_id):
        player = state.players[player_id]
        hand = player.hand

        if not hand:
            return 0.0

        alive = self.count_support_in_cards(hand, "alive", state.heisenbergs_count)
        dead = self.count_support_in_cards(hand, "dead", state.heisenbergs_count)
        empty = self.count_support_in_cards(hand, "empty", state.heisenbergs_count)

        return max(alive, dead, empty) / len(hand)

    def score_claim(self, state, player_id, action):
        player = state.players[player_id]

        p_true = self.claim_truth_probability(
            state,
            player_id,
            action.amount,
            action.cat_type,
        )

        rank = claim_rank(action.amount, action.cat_type)

        proof_count = len(action.proof_indices)
        swap_count = len(action.swap_indices)

        own_support = self.count_support_in_cards(
            player.hand,
            action.cat_type,
            state.heisenbergs_count,
        )

        hand_quality_before = self.hand_quality(state, player_id)

        score = 0.0

        score += 3.0 * p_true
        score += 0.025 * rank
        score += 0.12 * own_support

        if proof_count > 0:
            proof_value = proof_count * max(0.0, 0.45 - p_true)
            proof_value += 0.03 * proof_count
            score += proof_value

        if swap_count > 0:
            if hand_quality_before < 0.45:
                score += 0.08 * swap_count
            elif hand_quality_before < 0.60:
                score += 0.03 * swap_count
            else:
                score -= 0.04 * swap_count

        if p_true < 0.20:
            score -= 2.5

        if p_true > 0.75:
            score += 0.25

        if len(state.active_players()) <= 2 and p_true < 0.35:
            score -= 1.0

        return score

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
        threshold = self.doubt_threshold

        if len(state.active_players()) == 2:
            threshold += 0.03

        if claim.amount >= state.total_cards_in_play():
            threshold -= 0.10

        return (p_false - threshold) * 4.0

    def score_physicat(self, state, player_id, action):
        player = state.players[player_id]
        physicat = player.physicat

        if physicat is None:
            return -999.0

        t = physicat.physicat_type
        hand_quality = self.hand_quality(state, player_id)

        score = 0.10

        if t == "newton":
            score = 0.35
            if hand_quality < 0.50:
                score += 0.20

        elif t == "albert":
            score = 0.10
            if hand_quality < 0.40:
                score += 0.45
            elif hand_quality < 0.55:
                score += 0.20
            else:
                score -= 0.10

        elif t == "sally":
            score = 0.10 + 0.02 * len(state.discard_pile)

        elif t == "cecilia":
            score = 0.25
            if state.current_claim and state.current_claim.cat_type == "alive":
                score += 0.20

        elif t == "maria":
            score = 0.25
            if state.current_claim and state.current_claim.cat_type == "empty":
                score += 0.25

        elif t == "marie":
            alive_proofs = 0
            for p in state.active_players():
                alive_proofs += sum(1 for c in p.proofs if c.cat_type == "alive")
                alive_proofs += p.physicat_proofs.get("alive", 0)
            score = 0.08 + 0.12 * alive_proofs

        elif t == "michael":
            dead_proofs = 0
            for p in state.active_players():
                dead_proofs += sum(1 for c in p.proofs if c.cat_type == "dead")
                dead_proofs += p.physicat_proofs.get("dead", 0)
            score = 0.08 + 0.12 * dead_proofs

        elif t == "richard":
            score = 0.15
            if state.current_claim is not None:
                score += 0.15

        elif t == "stephen":
            score = 0.10
            if state.current_claim is not None:
                score += 0.20

        elif t == "neil":
            score = 0.20 + 0.05 * len(state.face_up_physicats)

        return score