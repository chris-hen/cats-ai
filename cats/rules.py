from cats.actions import MakeClaimAction, DoubtAction, UsePhysicatAction
from itertools import combinations
from itertools import combinations

CLAIM_ORDER_TEXT = """
1A 1D 2A 2D 1E 3A 3D 4A 4D 2E
5A 5D 6A 6D 3E 7A 7D 8A 8D 4E
9A 9D 10A 10D 5E 11A 11D 12A 12D 6E
13A 13D 14A 14D 7E 15A 15D 16A 16D 8E
17A 17D 18A 18D 9E 19A 19D 20A 20D 10E
21A 21D 22A 22D 11E 23A 23D 24A 24D 12E
25A 25D 26A 26D 13E
"""


SYMBOL_TO_TYPE = {
    "A": "alive",
    "D": "dead",
    "E": "empty",
}


def parse_claim_token(token):
    amount = int(token[:-1])
    symbol = token[-1]
    cat_type = SYMBOL_TO_TYPE[symbol]

    return amount, cat_type


CLAIM_ORDER = [
    parse_claim_token(token)
    for token in CLAIM_ORDER_TEXT.split()
]

CLAIM_TO_INDEX = {
    claim: index
    for index, claim in enumerate(CLAIM_ORDER)
}


def claim_rank(amount: int, cat_type: str) -> int:
    return CLAIM_TO_INDEX[(amount, cat_type)]

def card_can_be_proof(card, claim_type):
    return card.cat_type == claim_type or card.cat_type == "heisenberg"


def proof_indices_are_legal(state, action):
    player = state.players[state.current_player]
    indices = action.proof_indices

    if len(indices) != len(set(indices)):
        return False

    for index in indices:
        if index < 0 or index >= len(player.hand):
            return False

        card = player.hand[index]

        if not card_can_be_proof(card, action.cat_type):
            return False

    return True


def possible_proof_index_sets(state, cat_type):
    player = state.players[state.current_player]

    valid_indices = [
        i
        for i, card in enumerate(player.hand)
        if card_can_be_proof(card, cat_type)
    ]

    proof_sets = [()]

    for size in range(1, len(valid_indices) + 1):
        for combo in combinations(valid_indices, size):
            proof_sets.append(combo)

    return proof_sets


def is_claim_higher(new_amount, new_type, current_claim):
    if current_claim is None:
        return True

    return claim_rank(new_amount, new_type) > claim_rank(
        current_claim.amount,
        current_claim.cat_type,
    )


def legal_claim_actions(state):
    actions = []
    total_cards = state.total_cards_in_play()

    for amount, cat_type in CLAIM_ORDER:
        if amount > total_cards:
            continue

        if not is_claim_higher(amount, cat_type, state.current_claim):
            continue

        for proof_indices in possible_proof_index_sets(state, cat_type):
            for swap_indices in possible_swap_index_sets_after_proofs(state, proof_indices):
                actions.append(
                    MakeClaimAction(
                        amount,
                        cat_type,
                        proof_indices=proof_indices,
                        swap_indices=swap_indices,
                    )
                )

    return actions


def has_any_higher_claim(state):
    for amount, cat_type in CLAIM_ORDER:
        if amount > state.total_cards_in_play():
            continue

        if is_claim_higher(amount, cat_type, state.current_claim):
            return True

    return False


def stephen_can_be_used(state):
    current_player = state.current_player
    next_player = state.next_active_player(current_player)

    if next_player is None:
        return False

    # Wenn noch kein Claim liegt, kann der nächste Spieler einfach claimen.
    if state.current_claim is None:
        return True

    # Wenn der nächste Spieler NICHT der Claimer ist, kann er doubten oder erhöhen.
    if state.current_claim.player_id != next_player:
        return True

    # Wenn Stephen im 2-Spieler-Spiel zurück zum Claimer führt,
    # darf dieser nicht sich selbst doubten.
    # Also muss es mindestens einen höheren Claim geben.
    return has_any_higher_claim(state)


def legal_actions(state):
    actions = []
    actions.extend(legal_claim_actions(state))

    # Der Claimer darf seinen eigenen Claim nicht doubten.
    if (
        state.current_claim is not None
        and state.current_claim.player_id != state.current_player
    ):
        actions.append(DoubtAction())

    player = state.players[state.current_player]

    if player.exists and player.active and player.physicat is not None and not player.physicat_used:
        physicat_type = player.physicat.physicat_type

        if physicat_type == "neil":
            for i, face_up in enumerate(state.face_up_physicats):
                if face_up.physicat_type == "neil":
                    continue

                if face_up.physicat_type == "albert" and len(state.deck) < len(player.hand):
                    continue

                actions.append(UsePhysicatAction(target_index=i))
                
        elif physicat_type == "stephen":
            if stephen_can_be_used(state):
                actions.append(UsePhysicatAction())

        elif physicat_type == "albert":
            if len(state.deck) >= len(player.hand):
                actions.append(UsePhysicatAction())

        else:
            actions.append(UsePhysicatAction())

    return actions

def possible_swap_index_sets_after_proofs(state, proof_indices):
    player = state.players[state.current_player]

    remaining_indices = [
        i for i in range(len(player.hand))
        if i not in proof_indices
    ]

    max_swaps = min(len(proof_indices), len(remaining_indices), len(state.deck))

    swap_sets = [()]

    for size in range(1, max_swaps + 1):
        for combo in combinations(remaining_indices, size):
            swap_sets.append(combo)

    return swap_sets