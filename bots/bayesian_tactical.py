from cats.actions import MakeClaimAction, DoubtAction, UsePhysicatAction
from cats.rules import legal_actions, claim_rank
from bots.bayesian_v4 import BayesianV4Bot
from bots.bayesian_base import hypergeom_at_least


class BayesianTacticalBot(BayesianV4Bot):
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
            "claim_push_attempts": 0,
            "claim_push_successes": 0,
            "proof_swap_attempts": 0,
            "proof_swap_successes": 0,
            "physicat_uses": 0,
        }

        self.pending_kill_tactic = None
        self.pending_claim_push = False
        self.pending_proof_swap = False

    def observe(self, info):
        if info is None:
            return

        event = info.get("event")

        if event == "use_physicat":
            self.tactical_stats["physicat_uses"] += 1

        if event == "claim":
            proof_cards = info.get("proof_cards", [])
            swap_cards = info.get("swap_cards", [])

            if proof_cards and swap_cards:
                self.pending_proof_swap = True
                self.tactical_stats["proof_swap_attempts"] += 1

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

            if self.pending_claim_push:
                if claim_was_true:
                    self.tactical_stats["claim_push_successes"] += 1

                self.pending_claim_push = False

            if self.pending_proof_swap:
                if claim_was_true:
                    self.tactical_stats["proof_swap_successes"] += 1

                self.pending_proof_swap = False

    def choose_action(self, state):
        actions = legal_actions(state)

        if not actions:
            raise RuntimeError("No legal actions available")

        player_id = state.current_player

        kill_setup = self.find_kill_setup_action(state, player_id, actions)
        if kill_setup is not None:
            return kill_setup

        push_setup = self.find_claim_push_setup_action(state, player_id, actions)
        if push_setup is not None:
            return push_setup

        return super().choose_action(state)

    def find_use_physicat_action(self, actions, target_index=None):
        for action in actions:
            if isinstance(action, UsePhysicatAction):
                if target_index is None or action.target_index == target_index:
                    return action
        return None

    def current_player_physicat_type(self, state, player_id):
        player = state.players[player_id]

        if player.physicat is None or player.physicat_used:
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

        if physicat_type == "michael" and claim.cat_type == "dead":
            after_p_true = self.estimate_claim_probability_after_removing_proofs(
                state,
                player_id,
                claim.amount,
                claim.cat_type,
                removed_type="dead",
            )

            if current_p_true > 0.25 and after_p_true < 0.35:
                action = self.find_use_physicat_action(actions)
                if action is not None:
                    self.register_kill_tactic("michael")
                    return action

        if physicat_type == "richard":
            without_h = self.claim_truth_probability_custom_heisenberg(
                state,
                player_id,
                claim.amount,
                claim.cat_type,
                heisenbergs_count=False,
            )

            if current_p_true - without_h > 0.20 and without_h < 0.40:
                action = self.find_use_physicat_action(actions)
                if action is not None:
                    self.register_kill_tactic("richard")
                    return action

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
                    if after_p_true < 0.35:
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

    def register_claim_push(self):
        self.pending_claim_push = True
        self.tactical_stats["claim_push_attempts"] += 1

    def find_claim_push_setup_action(self, state, player_id, actions):
        physicat_type = self.current_player_physicat_type(state, player_id)

        if physicat_type is None:
            return None

        player = state.players[player_id]
        hand_quality = self.hand_quality(state, player_id)

        if physicat_type == "newton":
            if len(state.deck) >= 2 and hand_quality < 0.65:
                action = self.find_use_physicat_action(actions)
                if action is not None:
                    self.register_claim_push()
                    return action

        if physicat_type == "albert":
            if len(state.deck) >= len(player.hand) and hand_quality < 0.40:
                action = self.find_use_physicat_action(actions)
                if action is not None:
                    self.register_claim_push()
                    return action

        if physicat_type == "cecilia":
            best_alive = self.best_claim_probability_for_type(state, player_id, "alive")
            if best_alive > 0.45:
                action = self.find_use_physicat_action(actions)
                if action is not None:
                    self.register_claim_push()
                    return action

        if physicat_type == "maria":
            best_empty = self.best_claim_probability_for_type(state, player_id, "empty")
            if best_empty > 0.35:
                action = self.find_use_physicat_action(actions)
                if action is not None:
                    self.register_claim_push()
                    return action

        if physicat_type == "sally":
            if len(state.discard_pile) >= 3 and not player.saw_full_discard:
                return self.find_use_physicat_action(actions)

        if physicat_type == "stephen":
            if state.current_claim is not None:
                return self.find_use_physicat_action(actions)

        if physicat_type == "neil":
            priority = {
                "newton": 5,
                "richard": 4,
                "marie": 4,
                "michael": 4,
                "cecilia": 3,
                "maria": 3,
                "albert": 2,
                "sally": 1,
                "stephen": 1,
            }

            best_index = None
            best_score = -1

            for i, face_up in enumerate(state.face_up_physicats):
                score = priority.get(face_up.physicat_type, 0)

                if score > best_score:
                    best_score = score
                    best_index = i

            if best_index is not None and best_score >= 3:
                action = self.find_use_physicat_action(actions, target_index=best_index)
                if action is not None:
                    self.register_claim_push()
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

    def best_claim_probability_for_type(self, state, player_id, cat_type):
        actions = legal_actions(state)
        best = 0.0

        for action in actions:
            if not isinstance(action, MakeClaimAction):
                continue

            if action.cat_type != cat_type:
                continue

            p_true = self.claim_truth_probability(
                state,
                player_id,
                action.amount,
                action.cat_type,
            )

            best = max(best, p_true)

        return best

    def score_claim(self, state, player_id, action):
        base_score = super().score_claim(state, player_id, action)

        p_true = self.claim_truth_probability(
            state,
            player_id,
            action.amount,
            action.cat_type,
        )

        rank = claim_rank(action.amount, action.cat_type)
        proof_count = len(action.proof_indices)
        swap_count = len(action.swap_indices)

        score = base_score

        if p_true > 0.70:
            score += 0.015 * rank

        if 0.30 <= p_true <= 0.55:
            if proof_count == 1 and swap_count <= 1:
                if self.hand_quality(state, player_id) < 0.50:
                    score += 0.20

        if proof_count > 2:
            score -= 0.50 * (proof_count - 2)

        if swap_count > 1 and self.hand_quality(state, player_id) >= 0.45:
            score -= 0.30 * (swap_count - 1)

        player = state.players[player_id]

        if player.physicat_proofs.get(action.cat_type, 0) > 0:
            score += 0.20

        if len(state.active_players()) <= 2:
            if proof_count > 0 or swap_count > 0:
                score -= 0.20

        return score

    def score_doubt(self, state, player_id):
        score = super().score_doubt(state, player_id)

        claim = state.current_claim

        if claim is None:
            return score

        p_true = self.claim_truth_probability(
            state,
            player_id,
            claim.amount,
            claim.cat_type,
        )

        if not state.heisenbergs_count and p_true < 0.45:
            score += 0.25

        visible_support = 0
        for p in state.active_players():
            for card in p.proofs:
                if card.supports_claim(claim.cat_type, state.heisenbergs_count):
                    visible_support += 1

        if visible_support == 0 and p_true < 0.45:
            score += 0.15

        return score

    def score_physicat(self, state, player_id, action):
        player = state.players[player_id]
        physicat = player.physicat

        if physicat is None:
            return -999.0

        t = physicat.physicat_type
        hand_quality = self.hand_quality(state, player_id)

        score = super().score_physicat(state, player_id, action)

        if t == "newton":
            score += 0.30
            if hand_quality < 0.50:
                score += 0.20

        elif t == "albert":
            if hand_quality < 0.40:
                score += 0.45
            else:
                score -= 0.25

        elif t in ["marie", "michael", "richard"]:
            if state.current_claim is not None:
                score += 0.35

        elif t in ["cecilia", "maria"]:
            score += 0.20

        elif t == "sally":
            score += 0.02 * len(state.discard_pile)

        elif t == "neil":
            score += 0.05 * len(state.face_up_physicats)

        elif t == "stephen":
            if state.current_claim is not None:
                score += 0.20

        return score