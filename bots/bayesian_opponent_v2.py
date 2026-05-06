from collections import defaultdict

from bots.bayesian_v1 import BayesianV1Bot
from cats.rules import claim_rank


def smoothed_rate(successes, attempts, prior_rate=0.5, prior_weight=4):
    return (successes + prior_rate * prior_weight) / (attempts + prior_weight)


class BayesianOpponentV2Bot(BayesianV1Bot):
    """
    OpponentV2:
    - basiert auf BayesianV1, dem bisher stärksten Bot
    - hält Proofs/Swaps weiterhin stark zurück
    - nutzt Gegnerinformationen für:
        1. Doubt-Entscheidungen
        2. Riskante Claims gegen gute/aggressive Doubter
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
        super().__init__(
            doubt_threshold=doubt_threshold,
            claim_confidence_weight=claim_confidence_weight,
            claim_rank_weight=claim_rank_weight,
            proof_penalty=proof_penalty,
            swap_penalty=swap_penalty,
            physicat_bonus=physicat_bonus,
            random_noise=random_noise,
        )

        self.stats = defaultdict(lambda: {
            "claims": 0,
            "checked_claims": 0,
            "true_claims": 0,
            "false_claims": 0,

            "doubts": 0,
            "correct_doubts": 0,
            "wrong_doubts": 0,

            "proof_cards": 0,
            "claims_with_proof": 0,
            "swaps": 0,

            "claim_rank_total": 0,
            "claim_rank_jumps": 0,
        })

        self.last_claim_rank_by_player = {}

    def observe(self, info):
        if info is None:
            return

        event = info.get("event")

        if event == "claim":
            player = info["player"]
            claim = info["claim"]
            proof_cards = info.get("proof_cards", [])
            swap_cards = info.get("swap_cards", [])

            rank = claim_rank(claim.amount, claim.cat_type)

            s = self.stats[player]
            s["claims"] += 1
            s["claim_rank_total"] += rank

            if player in self.last_claim_rank_by_player:
                jump = max(0, rank - self.last_claim_rank_by_player[player])
                s["claim_rank_jumps"] += jump

            self.last_claim_rank_by_player[player] = rank

            if proof_cards:
                s["claims_with_proof"] += 1
                s["proof_cards"] += len(proof_cards)

            if swap_cards:
                s["swaps"] += len(swap_cards)

        elif event == "doubt":
            claimer = info["claimer"]
            doubter = info["doubter"]
            claim_was_true = info["claim_was_true"]

            self.stats[doubter]["doubts"] += 1

            self.stats[claimer]["checked_claims"] += 1

            if claim_was_true:
                self.stats[claimer]["true_claims"] += 1
                self.stats[doubter]["wrong_doubts"] += 1
            else:
                self.stats[claimer]["false_claims"] += 1
                self.stats[doubter]["correct_doubts"] += 1

    def bluff_rate(self, player_id):
        s = self.stats[player_id]
        return smoothed_rate(
            s["false_claims"],
            s["checked_claims"],
            prior_rate=0.25,
            prior_weight=6,
        )

    def honesty_rate(self, player_id):
        s = self.stats[player_id]
        return smoothed_rate(
            s["true_claims"],
            s["checked_claims"],
            prior_rate=0.75,
            prior_weight=6,
        )

    def doubt_accuracy(self, player_id):
        s = self.stats[player_id]
        attempts = s["correct_doubts"] + s["wrong_doubts"]
        return smoothed_rate(
            s["correct_doubts"],
            attempts,
            prior_rate=0.5,
            prior_weight=4,
        )

    def doubt_aggression(self, player_id):
        s = self.stats[player_id]
        return smoothed_rate(
            s["doubts"],
            s["claims"] + s["doubts"],
            prior_rate=0.20,
            prior_weight=8,
        )

    def proof_tendency(self, player_id):
        s = self.stats[player_id]
        return smoothed_rate(
            s["claims_with_proof"],
            s["claims"],
            prior_rate=0.10,
            prior_weight=8,
        )

    def avg_claim_rank(self, player_id):
        s = self.stats[player_id]
        if s["claims"] <= 0:
            return 0.0
        return s["claim_rank_total"] / s["claims"]

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
        claimer_id = claim.player_id

        bluff = self.bluff_rate(claimer_id)
        honesty = self.honesty_rate(claimer_id)
        proofiness = self.proof_tendency(claimer_id)

        adjusted_p_false = p_false

        # Spieler mit falschen überprüften Claims eher doubten.
        adjusted_p_false += 0.35 * (bluff - 0.25)

        # Sehr ehrliche Spieler weniger doubten.
        adjusted_p_false -= 0.20 * (honesty - 0.75)

        # Wer viel proofed, ist tendenziell glaubwürdiger / lesbarer.
        adjusted_p_false -= 0.10 * proofiness

        return (adjusted_p_false - self.doubt_threshold) * 4.0

    def score_claim(self, state, player_id, action):
        p_true = self.claim_truth_probability(
            state,
            player_id,
            action.amount,
            action.cat_type,
        )

        rank = claim_rank(action.amount, action.cat_type)

        score = 0.0

        # V1-Kern
        score += self.claim_confidence_weight * p_true
        score += self.claim_rank_weight * rank

        score -= self.proof_penalty * len(action.proof_indices)
        score -= self.swap_penalty * len(action.swap_indices)

        if p_true < 0.25:
            score -= 2.0

        next_player_id = state.next_active_player(player_id)

        if next_player_id is not None:
            next_doubt_aggression = self.doubt_aggression(next_player_id)
            next_doubt_accuracy = self.doubt_accuracy(next_player_id)

            claim_risk = 1.0 - p_true

            # Gegen aggressive UND korrekte Doubter keine dünnen Claims.
            danger = claim_risk * next_doubt_aggression * next_doubt_accuracy
            score -= 0.8 * danger

            # Gegen passive Doubter darf man minimal mehr Druck machen.
            if next_doubt_aggression < 0.15 and p_true > 0.35:
                score += 0.08 * claim_risk

        return score