from cats.actions import UsePhysicatAction
from cats.rules import legal_actions
from bots.bayesian_v4 import BayesianV4Bot
from bots.bayesian_base import hypergeom_at_least


class BayesianTacticalV2Bot(BayesianV4Bot):
    """
    TacticalBot v2:
    - basiert auf BayesianV4
    - nutzt NUR die bisher erfolgreichen Kill-Taktiken
    - entfernt Claim-Push und Proof-Swap-Logik
    - Ziel: V4/V1-Stabilität + gezielte Physicat-Kills

    Taktiken:
    - Richard + Doubt
    - Marie + Doubt gegen Alive-Claims
    - Michael + Doubt gegen Dead-Claims
    - Neil kopiert Richard/Marie/Michael, wenn sinnvoll
    """

    def __init__(self):
        super().__init__(
            doubt_threshold=0.42,
            claim_confidence_weight=3.0,
            claim_rank_weight=0.025,
            physicat_bonus=0.15,
            random_noise=0.01,
        )

        self.tactical_stats = {
            "kill_setup_attempts": 0,
            "kill_setup_successes": 0,
            "richard_kill_attempts": 0,
            "richard_kill_successes": 0,
            "marie_kill_attempts": 0,
            "marie_kill_successes": 0,
            "michael_kill_attempts": 0,
            "michael_kill_successes": 0,
            "physicat_uses": 0,
        }

        self.pending_kill_tactic = None

    def observe(self, info):
        if info is None:
            return

        event = info.get("event")

        if event == "use_physicat":
            self.tactical_stats["physicat_uses"] += 1

        if event == "doubt":
            claim_was_true = info["claim_was_true"]

            if self.pending_kill_tactic is not None:
                if not claim_was_true:
                    self.tactical_stats["kill_setup_successes"] += 1

                    if self.pending_kill_tactic == "richard":
                        self.tactical_stats["richard_kill_successes"] += 1
                    elif self.pending_kill_tactic == "marie":
                        self.tactical_stats["marie_kill_successes"] += 1
                    elif self.pending_kill_tactic == "michael":
                        self.tactical_stats["michael_kill_successes"] += 1

                self.pending_kill_tactic = None

    def choose_action(self, state):
        actions = legal_actions(state)

        if not actions:
            raise RuntimeError("No legal actions available")

        player_id = state.current_player

        kill_setup = self.find_kill_setup_action(state, player_id, actions)

        if kill_setup is not None:
            return kill_setup

        return super().choose_action(state)

    def find_use_physicat_action(self, actions, target_index=None):
        for action in actions:
            if isinstance(action, UsePhysicatAction):
                if target_index is None or action.target_index == target_index:
                    return action

        return None

    def current_player_physicat_type(self, state, player_id):
        player = state.players[player_id]

        if player.physicat is None:
            return None

        if player.physicat_used:
            return None

        return player.physicat.physicat_type

    def register_kill_tactic(self, tactic_name):
        self.pending_kill_tactic = tactic_name

        self.tactical_stats["kill_setup_attempts"] += 1

        if tactic_name == "richard":
            self.tactical_stats["richard_kill_attempts"] += 1
        elif tactic_name == "marie":
            self.tactical_stats["marie_kill_attempts"] += 1
        elif tactic_name == "michael":
            self.tactical_stats["michael_kill_attempts"] += 1

    def find_kill_setup_action(self, state, player_id, actions):
        claim = state.current_claim

        if claim is None:
            return None

        physicat_type = self.current_player_physicat_type(state, player_id)

        if physicat_type is None:
            return None

        current_p_true = self.claim_truth_probability(
            state,
            player_id,
            claim.amount,
            claim.cat_type,
        )

        # Marie entfernt Alive-Proofs.
        if physicat_type == "marie" and claim.cat_type == "alive":
            after_p_true = self.estimate_claim_probability_after_removing_proofs(
                state,
                player_id,
                claim.amount,
                claim.cat_type,
                removed_type="alive",
            )

            if current_p_true > 0.25 and after_p_true < 0.35:
                action = self.find_use_physicat_action(actions)

                if action is not None:
                    self.register_kill_tactic("marie")
                    return action

        # Michael entfernt Dead-Proofs.
        if physicat_type == "michael" and claim.cat_type == "dead":
            after_p_true = self.estimate_claim_probability_after_removing_proofs(
                state,
                player_id,
                claim.amount,
                claim.cat_type,
                removed_type="dead",
            )

            # Michael war schwächer als Marie/Richard, daher etwas strenger.
            if current_p_true > 0.30 and after_p_true < 0.30:
                action = self.find_use_physicat_action(actions)

                if action is not None:
                    self.register_kill_tactic("michael")
                    return action

        # Richard: Heisenbergs zählen nicht mehr.
        if physicat_type == "richard":
            without_h = self.claim_truth_probability_custom_heisenberg(
                state,
                player_id,
                claim.amount,
                claim.cat_type,
                heisenbergs_count=False,
            )

            # Richard war stark, aber wir bleiben selektiv.
            if current_p_true - without_h > 0.20 and without_h < 0.40:
                action = self.find_use_physicat_action(actions)

                if action is not None:
                    self.register_kill_tactic("richard")
                    return action

        # Neil kopiert nur starke Kill-Physicats.
        if physicat_type == "neil":
            for i, face_up in enumerate(state.face_up_physicats):
                copied = face_up.physicat_type

                if copied == "marie" and claim.cat_type == "alive":
                    after_p_true = self.estimate_claim_probability_after_removing_proofs(
                        state,
                        player_id,
                        claim.amount,
                        claim.cat_type,
                        removed_type="alive",
                    )

                    if after_p_true < 0.35:
                        action = self.find_use_physicat_action(actions, target_index=i)

                        if action is not None:
                            self.register_kill_tactic("marie")
                            return action

                if copied == "michael" and claim.cat_type == "dead":
                    after_p_true = self.estimate_claim_probability_after_removing_proofs(
                        state,
                        player_id,
                        claim.amount,
                        claim.cat_type,
                        removed_type="dead",
                    )

                    if after_p_true < 0.30:
                        action = self.find_use_physicat_action(actions, target_index=i)

                        if action is not None:
                            self.register_kill_tactic("michael")
                            return action

                if copied == "richard":
                    without_h = self.claim_truth_probability_custom_heisenberg(
                        state,
                        player_id,
                        claim.amount,
                        claim.cat_type,
                        heisenbergs_count=False,
                    )

                    if current_p_true - without_h > 0.20 and without_h < 0.40:
                        action = self.find_use_physicat_action(actions, target_index=i)

                        if action is not None:
                            self.register_kill_tactic("richard")
                            return action

        return None

    def claim_truth_probability_custom_heisenberg(
        self,
        state,
        player_id,
        amount,
        cat_type,
        heisenbergs_count,
    ):
        known_support = 0
        player = state.players[player_id]

        for card in player.hand:
            if card.supports_claim(cat_type, heisenbergs_count):
                known_support += 1

        for p in state.active_players():
            for card in p.proofs:
                if card.supports_claim(cat_type, heisenbergs_count):
                    known_support += 1

            known_support += p.physicat_proofs.get(cat_type, 0)

        needed_from_hidden = amount - known_support

        if needed_from_hidden <= 0:
            return 1.0

        hidden_cards_in_play = 0

        for p in state.active_players():
            if p.player_id != player_id:
                hidden_cards_in_play += len(p.hand)

        pool_counts = self.unknown_pool_counts(state, player_id)
        population_size = sum(pool_counts.values())
        success_count = self.support_count_in_pool(
            pool_counts,
            cat_type,
            heisenbergs_count=heisenbergs_count,
        )

        return hypergeom_at_least(
            population_size=population_size,
            success_count=success_count,
            draws=hidden_cards_in_play,
            needed_successes=needed_from_hidden,
        )

    def estimate_claim_probability_after_removing_proofs(
        self,
        state,
        player_id,
        amount,
        cat_type,
        removed_type,
    ):
        known_support = 0
        player = state.players[player_id]

        for card in player.hand:
            if card.supports_claim(cat_type, state.heisenbergs_count):
                known_support += 1

        for p in state.active_players():
            for card in p.proofs:
                if card.cat_type == removed_type:
                    continue

                if card.supports_claim(cat_type, state.heisenbergs_count):
                    known_support += 1

            if removed_type != cat_type:
                known_support += p.physicat_proofs.get(cat_type, 0)

        needed_from_hidden = amount - known_support

        if needed_from_hidden <= 0:
            return 1.0

        hidden_cards_in_play = 0

        for p in state.active_players():
            if p.player_id != player_id:
                hidden_cards_in_play += len(p.hand)

        pool_counts = self.unknown_pool_counts(state, player_id)
        population_size = sum(pool_counts.values())
        success_count = self.support_count_in_pool(
            pool_counts,
            cat_type,
            heisenbergs_count=state.heisenbergs_count,
        )

        return hypergeom_at_least(
            population_size=population_size,
            success_count=success_count,
            draws=hidden_cards_in_play,
            needed_successes=needed_from_hidden,
        )