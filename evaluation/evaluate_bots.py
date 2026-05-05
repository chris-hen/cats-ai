from collections import defaultdict

from cats.env import CatsEnv
from cats.rules import legal_actions
from cats.actions import DoubtAction
from bots.random_bot import RandomBot
from bots.bayesian_bot import BayesianHeuristicBot


def play_game_with_bot_factories(bot_factories, verbose=False, max_turns=500):
    num_players = len(bot_factories)
    env = CatsEnv(num_players=num_players)
    bots = [factory() for factory in bot_factories]

    state = env.reset()
    turns = 0
    forced_doubts = 0

    while not state.is_game_over() and turns < max_turns:
        player_id = state.current_player
        legal = legal_actions(state)

        if not legal:
            action = DoubtAction()
            forced_doubts += 1
        else:
            action = bots[player_id].choose_action(state)

        if verbose:
            print(f"Turn {turns}: P{player_id} -> {action}")

        state, reward, done, info = env.step(action)
        turns += 1

    return {
        "winner": state.winner(),
        "turns": turns,
        "forced_doubts": forced_doubts,
        "max_turn_reached": turns >= max_turns,
    }


def print_results(title, results, total_games):
    print()
    print(f"=== {title} ===")
    print(f"Total games: {total_games}")
    print()

    for bot_name, stats in results.items():
        games = stats["games"]
        wins = stats["wins"]
        avg_turns = stats["turns"] / games
        winrate = wins / games

        print(bot_name)
        print(f"  Games:          {games}")
        print(f"  Wins:           {wins}")
        print(f"  Winrate:        {winrate:.3%}")
        print(f"  AvgTurns:       {avg_turns:.2f}")
        print(f"  ForcedDoubts:   {stats['forced_doubts']}")
        print(f"  MaxTurnReached: {stats['max_turn_reached']}")
        print()


def new_stats():
    return {
        "games": 0,
        "wins": 0,
        "turns": 0,
        "forced_doubts": 0,
        "max_turn_reached": 0,
    }


def add_result(results, bot_name, seat, winner, result):
    results[bot_name]["games"] += 1
    results[bot_name]["turns"] += result["turns"]
    results[bot_name]["forced_doubts"] += result["forced_doubts"]
    results[bot_name]["max_turn_reached"] += int(result["max_turn_reached"])

    if seat == winner:
        results[bot_name]["wins"] += 1


def evaluate_one_bayesian_vs_random(num_games_per_seat=1000, num_players=4):
    results = defaultdict(new_stats)

    for bayesian_seat in range(num_players):
        print(f"Evaluating Bayesian seat {bayesian_seat}...")

        for _ in range(num_games_per_seat):
            factories = []

            for seat in range(num_players):
                if seat == bayesian_seat:
                    factories.append(BayesianHeuristicBot)
                else:
                    factories.append(RandomBot)

            result = play_game_with_bot_factories(factories)
            winner = result["winner"]

            for seat, factory in enumerate(factories):
                bot_name = (
                    "BayesianBot"
                    if factory is BayesianHeuristicBot
                    else "RandomBot"
                )
                add_result(results, bot_name, seat, winner, result)

    print_results(
        title="1 BayesianBot vs 3 RandomBots",
        results=results,
        total_games=num_games_per_seat * num_players,
    )


def evaluate_all_bayesian(num_games=1000, num_players=4):
    results = defaultdict(new_stats)

    factories = [BayesianHeuristicBot for _ in range(num_players)]

    for i in range(num_games):
        if i % 100 == 0:
            print(f"All Bayesian game {i}/{num_games}...")

        result = play_game_with_bot_factories(factories)
        winner = result["winner"]

        for seat in range(num_players):
            bot_name = f"BayesianSeat{seat}"
            add_result(results, bot_name, seat, winner, result)

    print_results(
        title="BayesianBot vs BayesianBot",
        results=results,
        total_games=num_games,
    )


def bayesian_tight():
    return BayesianHeuristicBot(doubt_threshold=0.55)


def bayesian_normal():
    return BayesianHeuristicBot(doubt_threshold=0.42)


def bayesian_loose():
    return BayesianHeuristicBot(doubt_threshold=0.30)


def evaluate_bayesian_parameter_matchup(num_games_per_seat=500, num_players=4):
    variants = [
        ("Tight", bayesian_tight),
        ("Normal", bayesian_normal),
        ("Loose", bayesian_loose),
        ("Random", RandomBot),
    ]

    results = defaultdict(new_stats)

    for rotation in range(num_players):
        print(f"Parameter matchup rotation {rotation}...")

        rotated = variants[rotation:] + variants[:rotation]
        factories = [factory for _, factory in rotated]
        names = [name for name, _ in rotated]

        for _ in range(num_games_per_seat):
            result = play_game_with_bot_factories(factories)
            winner = result["winner"]

            for seat, name in enumerate(names):
                add_result(results, name, seat, winner, result)

    print_results(
        title="Bayesian Parameter Matchup",
        results=results,
        total_games=num_games_per_seat * num_players,
    )


if __name__ == "__main__":
    evaluate_one_bayesian_vs_random(
        num_games_per_seat=1000,
        num_players=4,
    )

    evaluate_all_bayesian(
        num_games=1000,
        num_players=4,
    )

    evaluate_bayesian_parameter_matchup(
        num_games_per_seat=500,
        num_players=4,
    )