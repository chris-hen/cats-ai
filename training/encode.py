from cats.game_state import CLAIM_TYPES, MAX_PLAYERS
from cats.actions import MakeClaimAction, DoubtAction
from cats.rules import CLAIM_ORDER, CLAIM_TO_INDEX


CARD_TYPES = ["alive", "dead", "empty", "heisenberg"]

DOUBT_ACTION_ID = len(CLAIM_ORDER)


def count_cards(cards):
    counts = {card_type: 0 for card_type in CARD_TYPES}

    for card in cards:
        counts[card.cat_type] += 1

    return counts


def encode_observation(state, player_id):
    """
    Sichtbarer Spielzustand für einen Spieler.
    Immer gepaddet auf MAX_PLAYERS = 6.
    """

    player = state.players[player_id]
    hand_counts = count_cards(player.hand)

    vector = []

    # Eigene Hand
    vector.append(hand_counts["alive"])
    vector.append(hand_counts["dead"])
    vector.append(hand_counts["empty"])
    vector.append(hand_counts["heisenberg"])

    # Allgemeiner Zustand
    vector.append(state.num_players)
    vector.append(state.round_number)
    vector.append(state.cards_per_player)
    vector.append(state.total_cards_in_play())
    vector.append(len(state.deck))
    vector.append(len(state.discard_pile))

    # Aktueller Claim
    if state.current_claim is None:
        vector.append(0)          # claim amount
        vector.extend([0, 0, 0])  # claim type one-hot
        vector.append(-1)         # claimer
    else:
        vector.append(state.current_claim.amount)

        for claim_type in CLAIM_TYPES:
            vector.append(1 if state.current_claim.cat_type == claim_type else 0)

        vector.append(state.current_claim.player_id)

    # Spieler existiert?
    for p in state.players:
        vector.append(1 if p.exists else 0)

    # Spieler aktiv?
    for p in state.players:
        vector.append(1 if p.active else 0)

    # Leben pro Spieler
    for p in state.players:
        vector.append(p.lives if p.exists else 0)

    # Handgrößen pro Spieler
    for p in state.players:
        vector.append(len(p.hand) if p.exists else 0)

    # Proof-Größen pro Spieler
    for p in state.players:
        vector.append(len(p.proofs) if p.exists else 0)

    # Sichtbare Proof-Karten pro Spieler
    for p in state.players:
        proof_counts = count_cards(p.proofs)

        vector.append(proof_counts["alive"])
        vector.append(proof_counts["dead"])
        vector.append(proof_counts["empty"])
        vector.append(proof_counts["heisenberg"])

    # Wer ist dran?
    for p in state.players:
        vector.append(1 if p.exists and p.player_id == state.current_player else 0)

    return vector


def encode_action(action):
    if isinstance(action, MakeClaimAction):
        return CLAIM_TO_INDEX[(action.amount, action.cat_type)]

    if isinstance(action, DoubtAction):
        return DOUBT_ACTION_ID

    raise ValueError(f"Unknown action: {action}")


def decode_action(action_id):
    if action_id == DOUBT_ACTION_ID:
        return DoubtAction()

    amount, cat_type = CLAIM_ORDER[action_id]
    return MakeClaimAction(amount, cat_type)