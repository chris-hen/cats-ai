from cats.env import CatsEnv
from bots.random_bot import RandomBot
from bots.bayesian_bot import BayesianHeuristicBot
from training.encode import encode_observation, encode_action


def safe_encode_action(action):
    try:
        return encode_action(action)
    except ValueError:
        return None


def play_one_game(num_players=4, bot_factory=RandomBot, verbose=False):
    env = CatsEnv(num_players=num_players)
    bots = [bot_factory() for _ in range(num_players)]

    state = env.reset()
    history = []
    turn = 0

    while not state.is_game_over():
        player_id = state.current_player
        action = bots[player_id].choose_action(state)

        observation = encode_observation(state, player_id)
        action_id = safe_encode_action(action)

        entry = {
            "turn": turn,
            "player_id": player_id,
            "round_number": state.round_number,
            "cards_per_player": state.cards_per_player,
            "observation": observation,
            "action": action,
            "action_id": action_id,
            "current_claim": state.current_claim,
            "active_players": state.active_player_ids(),
            "info": None,
        }

        state, reward, done, info = env.step(action)
        entry["info"] = info

        history.append(entry)

        if verbose:
            print(f"Turn {turn}: P{player_id} -> {action}")

            if info["event"] == "doubt":
                print("DOUBT RESULT:", info)
                print("Active players:", state.active_player_ids())
                print()

        turn += 1

    return {
        "winner": state.winner(),
        "turns": turn,
        "history": history,
    }


def play_many_games(num_games=100, num_players=4, bot_factory=RandomBot, verbose=False):
    results = []

    for _ in range(num_games):
        results.append(
            play_one_game(
                num_players=num_players,
                bot_factory=bot_factory,
                verbose=verbose,
            )
        )

    return results


def play_bayesian_game(num_players=4, verbose=False):
    return play_one_game(
        num_players=num_players,
        bot_factory=BayesianHeuristicBot,
        verbose=verbose,
    )