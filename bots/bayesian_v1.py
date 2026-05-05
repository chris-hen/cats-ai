from bots.bayesian_base import BayesianBaseBot
from cats.rules import claim_rank


class BayesianV1Bot(BayesianBaseBot):
    """
    v1:
    - starke Bayesian-Wahrscheinlichkeit
    - Proofs und Swaps werden pauschal bestraft
    - Physicats mit festen Boni
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
        super().__init__(random_noise=random_noise)

        self.doubt_threshold = doubt_threshold
        self.claim_confidence_weight = claim_confidence_weight
        self.claim_rank_weight = claim_rank_weight
        self.proof_penalty = proof_penalty
        self.swap_penalty = swap_penalty
        self.physicat_bonus = physicat_bonus

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
        score += self.claim_confidence_weight * p_true
        score += self.claim_rank_weight * rank

        score -= self.proof_penalty * len(action.proof_indices)
        score -= self.swap_penalty * len(action.swap_indices)

        if p_true < 0.25:
            score -= 2.0

        return score

    def score_physicat(self, state, player_id, action):
        player = state.players[player_id]
        physicat = player.physicat

        if physicat is None:
            return -999.0

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

        return self.physicat_bonus + bonuses.get(physicat.physicat_type, 0.0)