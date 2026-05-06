from cats.game_state import GameState, Claim
from cats.actions import MakeClaimAction, DoubtAction, UsePhysicatAction
from cats.rules import legal_actions


class CatsEnv:
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.state = GameState(num_players=num_players)

    def reset(self):
        self.state = GameState(num_players=self.num_players)
        self.state.setup_round()
        return self.state

    def step(self, action):
        if action not in legal_actions(self.state):
            raise ValueError(f"Illegal action: {action}")

        if isinstance(action, MakeClaimAction):
            return self._make_claim(action)

        if isinstance(action, DoubtAction):
            return self._doubt()

        if isinstance(action, UsePhysicatAction):
            return self._use_physicat(action)

        raise ValueError(f"Unknown action: {action}")

    def _card_can_be_proof(self, card, claim_type):
        return card.cat_type == claim_type or card.cat_type == "heisenberg"

    def _validate_claim_action(self, action):
        player = self.state.players[self.state.current_player]

        proof_indices = tuple(action.proof_indices)
        swap_indices = tuple(action.swap_indices)

        if len(proof_indices) != len(set(proof_indices)):
            raise ValueError("Duplicate proof indices")

        if len(swap_indices) != len(set(swap_indices)):
            raise ValueError("Duplicate swap indices")

        if set(proof_indices) & set(swap_indices):
            raise ValueError("A card cannot be both proof and swapped")

        for index in proof_indices:
            if index < 0 or index >= len(player.hand):
                raise ValueError(f"Invalid proof index: {index}")

            card = player.hand[index]

            if not self._card_can_be_proof(card, action.cat_type):
                raise ValueError(
                    f"Invalid proof card {card} for claim type {action.cat_type}"
                )

        if swap_indices and not proof_indices:
            raise ValueError("Cannot swap without showing proof")

        if len(swap_indices) > len(proof_indices):
            raise ValueError("Cannot swap more cards than shown as proof")

        for index in swap_indices:
            if index < 0 or index >= len(player.hand):
                raise ValueError(f"Invalid swap index: {index}")

        if len(swap_indices) > len(self.state.deck):
            raise ValueError("Not enough cards in deck to swap")

    def _make_claim(self, action):
        self._validate_claim_action(action)

        player_id = self.state.current_player
        player = self.state.players[player_id]

        proof_cards = []
        swap_cards = []

        proof_index_set = set(action.proof_indices)
        swap_index_set = set(action.swap_indices)

        all_remove_indices = sorted(proof_index_set | swap_index_set, reverse=True)
        removed_by_index = {}

        for index in all_remove_indices:
            removed_by_index[index] = player.hand.pop(index)

        for index in sorted(proof_index_set):
            proof_cards.append(removed_by_index[index])

        for index in sorted(swap_index_set):
            swap_cards.append(removed_by_index[index])

        player.proofs.extend(proof_cards)

        self.state.discard_pile.extend(swap_cards)
        player.known_discard.extend(swap_cards)

        drawn_cards = self.state.deck.draw(len(swap_cards))
        player.hand.extend(drawn_cards)

        self.state.current_claim = Claim(
            amount=action.amount,
            cat_type=action.cat_type,
            player_id=player_id,
        )

        self.state.advance_turn()

        return self.state, 0, self.state.is_game_over(), {
            "event": "claim",
            "player": player_id,
            "claim": self.state.current_claim,
            "proof_cards": proof_cards,
            "swap_cards": swap_cards,
            "drawn_cards": drawn_cards,
        }

    def _remove_proofs_by_type(self, cat_type):
        removed_cards = []
        removed_physicat_proofs = 0

        for player in self.state.players:
            if not player.exists:
                continue

            remaining_proofs = []

            for card in player.proofs:
                if card.cat_type == cat_type:
                    removed_cards.append(card)
                else:
                    remaining_proofs.append(card)

            player.proofs = remaining_proofs

            removed_physicat_proofs += player.physicat_proofs.get(cat_type, 0)
            player.physicat_proofs[cat_type] = 0

            if cat_type == "alive":
                remaining_round_physicats = []

                for physicat in player.round_physicats:
                    if physicat.physicat_type == "cecilia":
                        self.state.face_up_physicats.append(physicat)
                    else:
                        remaining_round_physicats.append(physicat)

                player.round_physicats = remaining_round_physicats

        self.state.discard_pile.extend(removed_cards)

        # Entfernte Proof-Karten waren offen, also kennen alle existierenden Spieler sie.
        for player in self.state.players:
            if player.exists:
                player.known_discard.extend(removed_cards)

        return {
            "removed_cards": removed_cards,
            "removed_physicat_proofs": removed_physicat_proofs,
        }

    def _apply_physicat_effect(self, physicat, player_id):
        player = self.state.players[player_id]
        effect = None

        if physicat.physicat_type == "newton":
            drawn_cards = self.state.deck.draw(2)
            player.hand.extend(drawn_cards)

            effect = {
                "type": "draw_2",
                "drawn_cards": drawn_cards,
            }

        elif physicat.physicat_type == "albert":
            hand_size = len(player.hand)

            if len(self.state.deck) < hand_size:
                raise ValueError("Not enough cards in deck to swap entire hand")

            old_hand = list(player.hand)

            self.state.discard_pile.extend(old_hand)
            player.known_discard.extend(old_hand)

            player.hand = []

            drawn_cards = self.state.deck.draw(hand_size)
            player.hand.extend(drawn_cards)

            effect = {
                "type": "swap_entire_hand",
                "discarded_cards": old_hand,
                "drawn_cards": drawn_cards,
            }

        elif physicat.physicat_type == "marie":
            removed = self._remove_proofs_by_type("alive")

            effect = {
                "type": "remove_alive_proofs",
                **removed,
            }

        elif physicat.physicat_type == "michael":
            removed = self._remove_proofs_by_type("dead")

            effect = {
                "type": "remove_dead_proofs",
                **removed,
            }

        elif physicat.physicat_type == "richard":
            self.state.heisenbergs_count = False

            effect = {
                "type": "heisenbergs_do_not_count",
            }

        elif physicat.physicat_type == "cecilia":
            player.physicat_proofs["alive"] += 2

            effect = {
                "type": "physicat_proof",
                "cat_type": "alive",
                "amount": 2,
            }

        elif physicat.physicat_type == "maria":
            player.physicat_proofs["empty"] += 1

            effect = {
                "type": "physicat_proof",
                "cat_type": "empty",
                "amount": 1,
            }

        elif physicat.physicat_type == "sally":
            player.known_discard = list(self.state.discard_pile)
            player.saw_full_discard = True

            effect = {
                "type": "look_at_discard",
                "discard_cards": list(self.state.discard_pile),
            }

        elif physicat.physicat_type == "stephen":
            skipped_player = player_id
            self.state.advance_turn()

            effect = {
                "type": "skip_turn",
                "skipped_player": skipped_player,
                "next_player": self.state.current_player,
            }

        return effect

    def _use_physicat(self, action):
        player_id = self.state.current_player
        player = self.state.players[player_id]
        physicat = player.physicat

        if physicat is None:
            raise ValueError("Player has no Physicat")

        if player.physicat_used:
            raise ValueError("Physicat already used")

        if physicat.physicat_type == "neil":
            if action.target_index is None:
                raise ValueError("Neil needs a target_index")

            if action.target_index < 0 or action.target_index >= len(self.state.face_up_physicats):
                raise ValueError("Invalid Neil target_index")

            copied_physicat = self.state.face_up_physicats.pop(action.target_index)

            if copied_physicat.physicat_type == "neil":
                raise ValueError("Neil cannot copy Neil")

            player.physicat_used = True
            self.state.face_up_physicats.append(physicat)

            effect = self._apply_physicat_effect(copied_physicat, player_id)

            return self.state, 0, False, {
                "event": "use_physicat",
                "player": player_id,
                "physicat": physicat,
                "copied_physicat": copied_physicat,
                "effect": effect,
            }

        player.physicat_used = True

        if physicat.physicat_type in ["cecilia", "maria", "richard"]:
            player.round_physicats.append(physicat)
        else:
            self.state.face_up_physicats.append(physicat)

        effect = self._apply_physicat_effect(physicat, player_id)

        return self.state, 0, False, {
            "event": "use_physicat",
            "player": player_id,
            "physicat": physicat,
            "effect": effect,
        }

    def _doubt(self):
        doubter_id = self.state.current_player
        claim = self.state.current_claim
        claimer_id = claim.player_id

        actual_count = self.state.reveal_count_for_claim(
            claim,
            heisenbergs_count=self.state.heisenbergs_count,
        )

        if actual_count >= claim.amount:
            wrong_player = doubter_id
            right_player = claimer_id
            claim_was_true = True
        else:
            wrong_player = claimer_id
            right_player = doubter_id
            claim_was_true = False

        self.state.eliminate_or_damage_player(wrong_player)

        info = {
            "event": "doubt",
            "doubter": doubter_id,
            "claimer": claimer_id,
            "claim": claim,
            "actual_count": actual_count,
            "claim_was_true": claim_was_true,
            "wrong_player": wrong_player,
            "right_player": right_player,
        }

        if self.state.is_game_over():
            return self.state, 1, True, info

        self.state.start_next_round(starting_player=right_player)

        return self.state, 0, False, info