from bots.bayesian_v1 import BayesianV1Bot
from cats.rules import claim_rank


class BayesianV4Bot(BayesianV1Bot):
    """
    v4:
    - basiert strategisch auf V1
    - hält Informationen grundsätzlich zurück
    - zeigt Proofs nur sehr selektiv
    - swappt nur bei schwacher Hand
    """

    def __init__(
        self,
        doubt_threshold=0.42,
        claim_confidence_weight=3.0,
        claim_rank_weight=0.025,
        physicat_bonus=0.15,
        random_noise=0.01,
    ):
        super().__init__(
            doubt_threshold=doubt_threshold,
            claim_confidence_weight=claim_confidence_weight,
            claim_rank_weight=claim_rank_weight,
            proof_penalty=0.08,
            swap_penalty=0.04,
            physicat_bonus=physicat_bonus,
            random_noise=random_noise,
        )

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
        p_true = self.claim_truth_probability(
            state,
            player_id,
            action.amount,
            action.cat_type,
        )

        rank = claim_rank(action.amount, action.cat_type)

        proof_count = len(action.proof_indices)
        swap_count = len(action.swap_indices)

        score = 0.0

        # V1-Kernlogik
        score += self.claim_confidence_weight * p_true
        score += self.claim_rank_weight * rank

        # Sehr schlechte Claims vermeiden
        if p_true < 0.25:
            score -= 2.0

        # Informationszurückhaltung bleibt Default.
        score -= 0.10 * proof_count
        score -= 0.05 * swap_count

        # Proofs nur selektiv erlauben:
        # Ein einzelner Proof kann gut sein, wenn der Claim knapp/riskant,
        # aber nicht komplett unrealistisch ist.
        if proof_count == 1 and 0.25 <= p_true <= 0.45:
            score += 0.18

        # Viele Proofs fast immer bestrafen.
        if proof_count >= 2:
            score -= 0.20 * (proof_count - 1)

        # Swaps nur bei schlechter Hand leicht belohnen.
        hand_quality = self.hand_quality(state, player_id)

        if swap_count > 0:
            if hand_quality < 0.40:
                score += 0.08 * swap_count
            elif hand_quality < 0.55:
                score += 0.03 * swap_count
            else:
                score -= 0.08 * swap_count

        # Im Endgame noch weniger Information verschenken.
        if len(state.active_players()) <= 2:
            score -= 0.08 * proof_count
            score -= 0.05 * swap_count

        return score