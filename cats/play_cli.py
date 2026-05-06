from cats.env import CatsEnv
from cats.rules import legal_actions
from cats.actions import MakeClaimAction, DoubtAction, UsePhysicatAction
from bots.bayesian_v4 import BayesianV4Bot


TYPE_MAP = {
    "A": "alive",
    "D": "dead",
    "E": "empty",
}


def print_state(state, human_id=0):
    player = state.players[human_id]

    print("\n" + "=" * 60)
    print(f"Round {state.round_number}")
    print(f"Current player: P{state.current_player}")
    print(f"Current claim: {state.current_claim}")
    print(f"Cards in play: {state.total_cards_in_play()}")
    print(f"Face-up Physicats: {state.face_up_physicats}")

    print("\nPlayers:")
    for p in state.players:
        if p.exists:
            print(
                f"P{p.player_id}: "
                f"{'active' if p.active else 'out'}, "
                f"lives={p.lives}, "
                f"proofs={p.proofs}, "
                f"physicat={'used' if p.physicat_used else 'hidden'}"
            )

    print(f"\nYour hand:")
    for i, card in enumerate(player.hand):
        print(f"  {i}: {card}")

    print(f"Your Physicat: {player.physicat}, used={player.physicat_used}")


def parse_indices(text):
    text = text.strip()

    if not text:
        return ()

    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def choose_human_action(state):
    legal = legal_actions(state)

    while True:
        print("\nChoose action:")
        print("1) Make claim")

        option = 2

        can_doubt = any(isinstance(a, DoubtAction) for a in legal)
        if can_doubt:
            print(f"{option}) Doubt")
            doubt_option = option
            option += 1
        else:
            doubt_option = None

        physicat_actions = [a for a in legal if isinstance(a, UsePhysicatAction)]
        if physicat_actions:
            print(f"{option}) Use Physicat")
            physicat_option = option
        else:
            physicat_option = None

        choice = input("> ").strip()

        try:
            choice = int(choice)
        except ValueError:
            print("Invalid input.")
            continue

        if choice == 1:
            action = input_claim_action()

        elif doubt_option is not None and choice == doubt_option:
            action = DoubtAction()

        elif physicat_option is not None and choice == physicat_option:
            action = input_physicat_action(state, physicat_actions)

        else:
            print("Invalid choice.")
            continue

        if action in legal:
            return action

        print("\nIllegal action, try again.")
        print(f"You tried: {action}")


def input_claim_action():
    print("\nMake claim")

    amount = int(input("Amount: ").strip())

    raw_type = input("Type (A=alive, D=dead, E=empty): ").strip().upper()
    cat_type = TYPE_MAP[raw_type]

    proof_text = input("Proof indices, comma-separated, empty for none: ")
    proof_indices = parse_indices(proof_text)

    swap_text = input("Swap indices, comma-separated, empty for none: ")
    swap_indices = parse_indices(swap_text)

    return MakeClaimAction(
        amount=amount,
        cat_type=cat_type,
        proof_indices=proof_indices,
        swap_indices=swap_indices,
    )


def input_physicat_action(state, physicat_actions):
    player = state.players[state.current_player]
    physicat = player.physicat

    if physicat.physicat_type != "neil":
        return UsePhysicatAction()

    print("\nFace-up Physicats:")
    for i, face_up in enumerate(state.face_up_physicats):
        print(f"  {i}: {face_up}")

    target_index = int(input("Choose target index for Neil: ").strip())
    return UsePhysicatAction(target_index=target_index)


def print_doubt_result(info):
    print("\n" + "-" * 60)
    print("DOUBT RESULT")
    print(f"Claim: {info['claim']}")
    print(f"Actual count: {info['actual_count']}")
    print(f"Claim was true: {info['claim_was_true']}")
    print(f"Wrong player: P{info['wrong_player']}")
    print(f"Right player: P{info['right_player']}")
    print("-" * 60)


def play_cli(num_players=4):
    env = CatsEnv(num_players=num_players)
    state = env.reset()

    bots = {
        player_id: BayesianV4Bot()
        for player_id in range(1, num_players)
    }

    while not state.is_game_over():
        player_id = state.current_player

        if player_id == 0:
            print_state(state)
            action = choose_human_action(state)
        else:
            action = bots[player_id].choose_action(state)
            print(f"\nBot P{player_id} plays: {action}")

        state, reward, done, info = env.step(action)

        for bot in bots.values():
            if hasattr(bot, "observe"):
                bot.observe(info)

        if info and info.get("event") == "doubt":
            print_doubt_result(info)

    print("\nGame over!")
    print(f"Winner: P{state.winner()}")