from collections import defaultdict

from cats.env import CatsEnv
from cats.rules import legal_actions
from cats.actions import MakeClaimAction, DoubtAction, UsePhysicatAction

from bots.random_bot import RandomBot
from bots.bayesian_v1 import BayesianV1Bot
from bots.bayesian_v2 import BayesianV2Bot
from bots.bayesian_v3 import BayesianV3Bot
from bots.bayesian_opponent import BayesianOpponentBot


def play_game_with_bot_factories(bot_factories, verbose=False, max_turns=500):
    num_players = len(bot_factories)
    env = CatsEnv(num_players=num_players)
    bots = [factory() for factory in bot_factories]

    state = env.reset()
    turns = 0
    no_legal_actions = 0
    illegal_bot_actions = 0

    action_stats = [
        {
            "claims": 0,
            "doubts": 0,
            "proofs": 0,
            "swaps": 0,
            "physicat": 0,
        }
        for _ in range(num_players)
    ]

    while not state.is_game_over() and turns < max_turns:
        player_id = state.current_player
        legal = legal_actions(state)

        if not legal:
            no_legal_actions += 1
            break

        action = bots[player_id].choose_action(state)

        if action not in legal:
            illegal_bot_actions += 1
            action = legal[0]

        if isinstance(action, MakeClaimAction):
            action_stats[player_id]["claims"] += 1
            action_stats[player_id]["proofs"] += len(action.proof_indices)
            action_stats[player_id]["swaps"] += len(action.swap_indices)

        elif isinstance(action, DoubtAction):
            action_stats[player_id]["doubts"] += 1

        elif isinstance(action, UsePhysicatAction):
            action_stats[player_id]["physicat"] += 1

        if verbose:
            print(f"Turn {turns}: P{player_id} -> {action}")

        state, reward, done, info = env.step(action)

        for bot in bots:
            if hasattr(bot, "observe"):
                bot.observe(info)

        turns += 1

    return {
        "winner": state.winner(),
        "turns": turns,
        "no_legal_actions": no_legal_actions,
        "illegal_bot_actions": illegal_bot_actions,
        "max_turn_reached": turns >= max_turns,
        "action_stats": action_stats,
    }


def new_stats():
    return {
        "games": 0,
        "wins": 0,
        "turns": 0,
        "no_legal_actions": 0,
        "illegal_bot_actions": 0,
        "max_turn_reached": 0,
        "claims": 0,
        "doubts": 0,
        "proofs": 0,
        "swaps": 0,
        "physicat": 0,
    }


def add_result(results, bot_name, seat, winner, result):
    results[bot_name]["games"] += 1
    results[bot_name]["turns"] += result["turns"]
    results[bot_name]["no_legal_actions"] += result["no_legal_actions"]
    results[bot_name]["illegal_bot_actions"] += result["illegal_bot_actions"]
    results[bot_name]["max_turn_reached"] += int(result["max_turn_reached"])

    player_stats = result["action_stats"][seat]

    results[bot_name]["claims"] += player_stats["claims"]
    results[bot_name]["doubts"] += player_stats["doubts"]
    results[bot_name]["proofs"] += player_stats["proofs"]
    results[bot_name]["swaps"] += player_stats["swaps"]
    results[bot_name]["physicat"] += player_stats["physicat"]

    if seat == winner:
        results[bot_name]["wins"] += 1


def print_results(title, results, total_games):
    print()
    print(f"=== {title} ===")
    print(f"Total games: {total_games}")
    print()

    for bot_name, stats in results.items():
        games = stats["games"]
        wins = stats["wins"]
        winrate = wins / games if games else 0.0
        avg_turns = stats["turns"] / games if games else 0.0

        print(bot_name)
        print(f"  Games:          {games}")
        print(f"  Wins:           {wins}")
        print(f"  Winrate:        {winrate:.3%}")
        print(f"  AvgTurns:       {avg_turns:.2f}")
        print(f"  Claims/Game:    {stats['claims'] / games:.2f}")
        print(f"  Doubts/Game:    {stats['doubts'] / games:.2f}")
        print(f"  Proofs/Game:    {stats['proofs'] / games:.2f}")
        print(f"  Swaps/Game:     {stats['swaps'] / games:.2f}")
        print(f"  Physicat/Game:  {stats['physicat'] / games:.2f}")
        print(f"  NoLegalActions: {stats['no_legal_actions']}")
        print(f"  IllegalActions: {stats['illegal_bot_actions']}")
        print(f"  MaxTurnReached: {stats['max_turn_reached']}")
        print()


def evaluate_one_bot_vs_random(
    bot_name,
    bot_factory,
    num_games_per_seat=200,
    num_players=4,
):
    results = defaultdict(new_stats)

    for bot_seat in range(num_players):
        print(f"Evaluating {bot_name} seat {bot_seat}...")

        for _ in range(num_games_per_seat):
            factories = []

            for seat in range(num_players):
                factories.append(bot_factory if seat == bot_seat else RandomBot)

            result = play_game_with_bot_factories(factories)
            winner = result["winner"]

            for seat, factory in enumerate(factories):
                name = bot_name if seat == bot_seat else "RandomBot"
                add_result(results, name, seat, winner, result)

    print_results(
        title=f"1 {bot_name} vs 3 RandomBots",
        results=results,
        total_games=num_games_per_seat * num_players,
    )


def evaluate_self_play(
    bot_name,
    bot_factory,
    num_games=200,
    num_players=4,
):
    results = defaultdict(new_stats)
    factories = [bot_factory for _ in range(num_players)]

    for i in range(num_games):
        if i % 50 == 0:
            print(f"{bot_name} self-play game {i}/{num_games}...")

        result = play_game_with_bot_factories(factories)
        winner = result["winner"]

        for seat in range(num_players):
            add_result(results, f"{bot_name}Seat{seat}", seat, winner, result)

    print_results(
        title=f"{bot_name} self-play",
        results=results,
        total_games=num_games,
    )


def evaluate_one_bot_vs_baseline(
    challenger_name,
    challenger_factory,
    baseline_name,
    baseline_factory,
    num_games_per_seat=200,
    num_players=4,
):
    results = defaultdict(new_stats)

    for challenger_seat in range(num_players):
        print(f"{challenger_name} seat {challenger_seat} vs {baseline_name}...")

        for _ in range(num_games_per_seat):
            factories = []

            for seat in range(num_players):
                factories.append(
                    challenger_factory if seat == challenger_seat else baseline_factory
                )

            result = play_game_with_bot_factories(factories)
            winner = result["winner"]

            for seat, factory in enumerate(factories):
                name = challenger_name if seat == challenger_seat else baseline_name
                add_result(results, name, seat, winner, result)

    print_results(
        title=f"{challenger_name} vs 3 {baseline_name}",
        results=results,
        total_games=num_games_per_seat * num_players,
    )


def evaluate_bot_pool(num_games_per_rotation=200):
    variants = [
        ("Random", RandomBot),
        ("BayesianV1", BayesianV1Bot),
        ("BayesianV2", BayesianV2Bot),
        ("BayesianV3", BayesianV3Bot),
        ("OpponentBot", BayesianOpponentBot),
    ]

    num_players = 4
    results = defaultdict(new_stats)

    matchups = []

    for start in range(len(variants)):
        matchup = []

        for offset in range(num_players):
            matchup.append(variants[(start + offset) % len(variants)])

        matchups.append(matchup)

    for matchup_index, matchup in enumerate(matchups):
        print(f"Pool matchup {matchup_index + 1}/{len(matchups)}...")

        for rotation in range(num_players):
            rotated = matchup[rotation:] + matchup[:rotation]
            names = [name for name, _ in rotated]
            factories = [factory for _, factory in rotated]

            for _ in range(num_games_per_rotation):
                result = play_game_with_bot_factories(factories)
                winner = result["winner"]

                for seat, name in enumerate(names):
                    add_result(results, name, seat, winner, result)

    print_results(
        title="Bot Pool Evaluation",
        results=results,
        total_games=len(matchups) * num_players * num_games_per_rotation,
    )


if __name__ == "__main__":
    evaluate_one_bot_vs_random(
        bot_name="BayesianV1",
        bot_factory=BayesianV1Bot,
        num_games_per_seat=200,
        num_players=4,
    )

    evaluate_one_bot_vs_random(
        bot_name="BayesianV2",
        bot_factory=BayesianV2Bot,
        num_games_per_seat=200,
        num_players=4,
    )

    evaluate_one_bot_vs_baseline(
        challenger_name="BayesianV2",
        challenger_factory=BayesianV2Bot,
        baseline_name="BayesianV1",
        baseline_factory=BayesianV1Bot,
        num_games_per_seat=200,
        num_players=4,
    )

    evaluate_one_bot_vs_baseline(
        challenger_name="BayesianV3",
        challenger_factory=BayesianV3Bot,
        baseline_name="BayesianV1",
        baseline_factory=BayesianV1Bot,
        num_games_per_seat=200,
        num_players=4,
    )

    evaluate_one_bot_vs_baseline(
        challenger_name="OpponentBot",
        challenger_factory=BayesianOpponentBot,
        baseline_name="BayesianV1",
        baseline_factory=BayesianV1Bot,
        num_games_per_seat=200,
        num_players=4,
    )

    evaluate_bot_pool(
        num_games_per_rotation=200,
    )