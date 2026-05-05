from collections import defaultdict

from cats.env import CatsEnv
from cats.rules import legal_actions
from bots.bayesian_v2 import BayesianV2Bot


def make_bot_factory(bot_class, **params):
    def factory():
        return bot_class(**params)

    return factory


def new_stats():
    return {
        "games": 0,
        "wins": 0,
        "turns": 0,
        "no_legal_actions": 0,
        "illegal_bot_actions": 0,
        "max_turn_reached": 0,
    }


def add_result(results, name, seat, winner, result):
    results[name]["games"] += 1
    results[name]["turns"] += result["turns"]
    results[name]["no_legal_actions"] += result["no_legal_actions"]
    results[name]["illegal_bot_actions"] += result["illegal_bot_actions"]
    results[name]["max_turn_reached"] += int(result["max_turn_reached"])

    if seat == winner:
        results[name]["wins"] += 1


def play_game(bot_factories, max_turns=500):
    env = CatsEnv(num_players=len(bot_factories))
    bots = [factory() for factory in bot_factories]

    state = env.reset()
    turns = 0
    no_legal_actions = 0
    illegal_bot_actions = 0

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
    }


def tune_doubt_threshold(
    bot_class=BayesianV2Bot,
    thresholds=None,
    num_games_per_rotation=500,
    num_players=4,
):
    if thresholds is None:
        thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    variants = []

    for threshold in thresholds:
        name = f"{bot_class.__name__}(threshold={threshold:.2f})"
        factory = make_bot_factory(bot_class, doubt_threshold=threshold)
        variants.append((name, factory))

    if len(variants) < num_players:
        raise ValueError("Need at least as many variants as players.")

    results = defaultdict(new_stats)

    matchups = []

    for start in range(len(variants)):
        matchup = []

        for offset in range(num_players):
            matchup.append(variants[(start + offset) % len(variants)])

        matchups.append(matchup)

    game_counter = 0

    for matchup_index, matchup in enumerate(matchups):
        print(f"Matchup {matchup_index + 1}/{len(matchups)}...")

        for rotation in range(num_players):
            rotated = matchup[rotation:] + matchup[:rotation]
            names = [name for name, _ in rotated]
            factories = [factory for _, factory in rotated]

            for _ in range(num_games_per_rotation):
                result = play_game(factories)
                winner = result["winner"]

                for seat, name in enumerate(names):
                    add_result(results, name, seat, winner, result)

                game_counter += 1

    print()
    print("=== Bayesian Doubt Threshold Tuning ===")
    print(f"Bot class: {bot_class.__name__}")
    print(f"Players: {num_players}")
    print(f"Games per rotation: {num_games_per_rotation}")
    print(f"Total games: {game_counter}")
    print()

    rows = []

    for name, stats in results.items():
        games = stats["games"]
        wins = stats["wins"]
        winrate = wins / games if games else 0.0
        avg_turns = stats["turns"] / games if games else 0.0

        rows.append((winrate, name, games, wins, avg_turns, stats))

    rows.sort(reverse=True)

    for winrate, name, games, wins, avg_turns, stats in rows:
        print(name)
        print(f"  Games:          {games}")
        print(f"  Wins:           {wins}")
        print(f"  Winrate:        {winrate:.3%}")
        print(f"  AvgTurns:       {avg_turns:.2f}")
        print(f"  NoLegalActions: {stats['no_legal_actions']}")
        print(f"  IllegalActions: {stats['illegal_bot_actions']}")
        print(f"  MaxTurnReached: {stats['max_turn_reached']}")
        print()


if __name__ == "__main__":
    tune_doubt_threshold(
        bot_class=BayesianV2Bot,
        thresholds=[0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46],
        num_games_per_rotation=500,
        num_players=4,
    )