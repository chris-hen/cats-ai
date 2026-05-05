from collections import defaultdict
from bots.bayesian_v1 import BayesianV1Bot
from bots.bayesian_v2 import BayesianV2Bot
from bots.bayesian_v3 import BayesianV3Bot

class BayesianOpponentBot(BayesianV3Bot):
    """
    v4 / OpponentBot:
    - basiert auf v3
    - lernt innerhalb einer Partie einfache Gegnerstatistiken
    """

    def __init__(self):
        super().__init__()

        self.stats = defaultdict(lambda: {
            "claims": 0,
            "true_claims": 0,
            "false_claims": 0,
            "doubts": 0,
            "correct_doubts": 0,
            "wrong_doubts": 0,
        })

    def observe(self, info):
        if info is None:
            return

        event = info.get("event")

        if event == "claim":
            player = info["player"]
            self.stats[player]["claims"] += 1

        elif event == "doubt":
            claimer = info["claimer"]
            doubter = info["doubter"]
            claim_was_true = info["claim_was_true"]

            self.stats[doubter]["doubts"] += 1

            if claim_was_true:
                self.stats[claimer]["true_claims"] += 1
                self.stats[doubter]["wrong_doubts"] += 1
            else:
                self.stats[claimer]["false_claims"] += 1
                self.stats[doubter]["correct_doubts"] += 1

    def estimated_bluff_rate(self, player_id):
        s = self.stats[player_id]

        false_claims = s["false_claims"] + 1
        checked_claims = s["true_claims"] + s["false_claims"] + 4

        return false_claims / checked_claims

    def estimated_doubt_aggression(self, player_id):
        s = self.stats[player_id]

        doubts = s["doubts"] + 1
        opportunities_prior = 8

        return doubts / (s["claims"] + opportunities_prior)

    def estimated_doubt_accuracy(self, player_id):
        s = self.stats[player_id]

        correct = s["correct_doubts"] + 1
        total = s["correct_doubts"] + s["wrong_doubts"] + 3

        return correct / total

    def score_doubt(self, state, player_id):
        claim = state.current_claim

        if claim is None:
            return -999.0

        base_score = super().score_doubt(state, player_id)

        claimer_id = claim.player_id
        bluff_rate = self.estimated_bluff_rate(claimer_id)

        opponent_adjustment = (bluff_rate - 0.25) * 1.5

        return base_score + opponent_adjustment

    def score_claim(self, state, player_id, action):
        base_score = super().score_claim(state, player_id, action)

        next_player_id = state.next_active_player(player_id)

        if next_player_id is None:
            return base_score

        doubt_aggression = self.estimated_doubt_aggression(next_player_id)
        doubt_accuracy = self.estimated_doubt_accuracy(next_player_id)

        p_true = self.claim_truth_probability(
            state,
            player_id,
            action.amount,
            action.cat_type,
        )

        risk = (1.0 - p_true) * doubt_aggression * doubt_accuracy

        base_score -= 1.2 * risk

        if doubt_aggression < 0.15:
            base_score += 0.15 * (1.0 - p_true)

        return base_score