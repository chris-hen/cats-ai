from bots.bayesian_v1 import BayesianV1Bot
from cats.rules import claim_rank


class BayesianV2Bot(BayesianV1Bot):
    """
    v2:
    - basiert auf v1
    - doubt_threshold auf getunten Wert gesetzt
    - Proofs werden nicht mehr pauschal bestraft
    - Proofs können bei unsicheren Claims positiv sein
    - Swaps nur minimal bestraft
    """

    def __init__(
        self,
        doubt_threshold=0.34,
        claim_confidence_weight=3.0,
        claim_rank_weight=0.025,
        proof_penalty=0.0,
        swap_penalty=0.01,
        physicat_bonus=0.15,
        random_noise=0.01,
    ):
        super().__init__(
            doubt_threshold=doubt_threshold,
            claim_confidence_weight=claim_confidence_weight,
            claim_rank_weight=claim_rank_weight,
            proof_penalty=proof_penalty,
            swap_penalty=swap_penalty,
            physicat_bonus=physicat_bonus,
            random_noise=random_noise,
        )

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
        score += self.claim_confidence_weight * p_true
        score += self.claim_rank_weight * rank

        proof_bonus = proof_count * max(0.0, 0.35 - p_true)

        score += proof_bonus
        score += 0.02 * proof_count

        score -= self.swap_penalty * swap_count

        if p_true < 0.25:
            score -= 2.0

        return score