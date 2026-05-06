from collections import defaultdict

from cats.env import CatsEnv
from cats.rules import legal_actions, claim_rank
from cats.actions import MakeClaimAction, DoubtAction, UsePhysicatAction

from bots.random_bot import RandomBot
from bots.bayesian_v1 import BayesianV1Bot
from bots.bayesian_v4 import BayesianV4Bot
from bots.bayesian_tactical import BayesianTacticalBot
from bots.bayesian_tactical_v2 import BayesianTacticalV2Bot


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

            "claim_rank_sum": 0,
            "claim_amount_sum": 0,

            "got_doubted": 0,
            "checked_true_claims": 0,
            "checked_false_claims": 0,

            "correct_doubts": 0,
            "wrong_doubts": 0,

            "alive_claims": 0,
            "dead_claims": 0,
            "empty_claims": 0,
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

            action_stats[player_id]["claim_rank_sum"] += claim_rank(
                action.amount,
                action.cat_type,
            )
            action_stats[player_id]["claim_amount_sum"] += action.amount

            if action.cat_type == "alive":
                action_stats[player_id]["alive_claims"] += 1
            elif action.cat_type == "dead":
                action_stats[player_id]["dead_claims"] += 1
            elif action.cat_type == "empty":
                action_stats[player_id]["empty_claims"] += 1

        elif isinstance(action, DoubtAction):
            action_stats[player_id]["doubts"] += 1

        elif isinstance(action, UsePhysicatAction):
            action_stats[player_id]["physicat"] += 1

        if verbose:
            print(f"Turn {turns}: P{player_id} -> {action}")

        state, reward, done, info = env.step(action)

        if info is not None and info.get("event") == "doubt":
            doubter = info["doubter"]
            claimer = info["claimer"]
            claim_was_true = info["claim_was_true"]

            action_stats[claimer]["got_doubted"] += 1

            if claim_was_true:
                action_stats[claimer]["checked_true_claims"] += 1
                action_stats[doubter]["wrong_doubts"] += 1
            else:
                action_stats[claimer]["checked_false_claims"] += 1
                action_stats[doubter]["correct_doubts"] += 1

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
        "bot_objects": bots,
    }


def new_stats():
    return {
        "games": 0,
        "wins": 0,
        "turns": 0,

        "claims": 0,
        "doubts": 0,
        "proofs": 0,
        "swaps": 0,
        "physicat": 0,

        "claim_rank_sum": 0,
        "claim_amount_sum": 0,

        "got_doubted": 0,
        "checked_true_claims": 0,
        "checked_false_claims": 0,

        "correct_doubts": 0,
        "wrong_doubts": 0,

        "alive_claims": 0,
        "dead_claims": 0,
        "empty_claims": 0,

        "no_legal_actions": 0,
        "illegal_bot_actions": 0,
        "max_turn_reached": 0,

        "tactical": defaultdict(int),
    }


def add_result(results, bot_name, seat, winner, result):
    player_stats = result["action_stats"][seat]

    results[bot_name]["games"] += 1
    results[bot_name]["turns"] += result["turns"]

    results[bot_name]["claims"] += player_stats["claims"]
    results[bot_name]["doubts"] += player_stats["doubts"]
    results[bot_name]["proofs"] += player_stats["proofs"]
    results[bot_name]["swaps"] += player_stats["swaps"]
    results[bot_name]["physicat"] += player_stats["physicat"]

    results[bot_name]["claim_rank_sum"] += player_stats["claim_rank_sum"]
    results[bot_name]["claim_amount_sum"] += player_stats["claim_amount_sum"]

    results[bot_name]["got_doubted"] += player_stats["got_doubted"]
    results[bot_name]["checked_true_claims"] += player_stats["checked_true_claims"]
    results[bot_name]["checked_false_claims"] += player_stats["checked_false_claims"]

    results[bot_name]["correct_doubts"] += player_stats["correct_doubts"]
    results[bot_name]["wrong_doubts"] += player_stats["wrong_doubts"]

    results[bot_name]["alive_claims"] += player_stats["alive_claims"]
    results[bot_name]["dead_claims"] += player_stats["dead_claims"]
    results[bot_name]["empty_claims"] += player_stats["empty_claims"]

    results[bot_name]["no_legal_actions"] += result["no_legal_actions"]
    results[bot_name]["illegal_bot_actions"] += result["illegal_bot_actions"]
    results[bot_name]["max_turn_reached"] += int(result["max_turn_reached"])

    bot_object = result["bot_objects"][seat]

    if hasattr(bot_object, "tactical_stats"):
        for key, value in bot_object.tactical_stats.items():
            results[bot_name]["tactical"][key] += value

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

        claims = stats["claims"]
        doubts = stats["doubts"]
        checked_claims = stats["checked_true_claims"] + stats["checked_false_claims"]
        total_claim_type = stats["alive_claims"] + stats["dead_claims"] + stats["empty_claims"]

        claims_per_game = claims / games if games else 0.0
        doubts_per_game = doubts / games if games else 0.0
        proofs_per_game = stats["proofs"] / games if games else 0.0
        swaps_per_game = stats["swaps"] / games if games else 0.0
        physicat_per_game = stats["physicat"] / games if games else 0.0

        avg_claim_rank = stats["claim_rank_sum"] / claims if claims else 0.0
        avg_claim_amount = stats["claim_amount_sum"] / claims if claims else 0.0

        claim_truth_rate = (
            stats["checked_true_claims"] / checked_claims
            if checked_claims else 0.0
        )

        false_claim_rate = (
            stats["checked_false_claims"] / checked_claims
            if checked_claims else 0.0
        )

        doubt_accuracy = (
            stats["correct_doubts"] / doubts
            if doubts else 0.0
        )

        got_doubted_rate = (
            stats["got_doubted"] / claims
            if claims else 0.0
        )

        alive_rate = (
            stats["alive_claims"] / total_claim_type
            if total_claim_type else 0.0
        )

        dead_rate = (
            stats["dead_claims"] / total_claim_type
            if total_claim_type else 0.0
        )

        empty_rate = (
            stats["empty_claims"] / total_claim_type
            if total_claim_type else 0.0
        )

        print(bot_name)
        print(f"  Games:            {games}")
        print(f"  Wins:             {wins}")
        print(f"  Winrate:          {winrate:.3%}")
        print(f"  AvgTurns:         {avg_turns:.2f}")

        print(f"  Claims/Game:      {claims_per_game:.2f}")
        print(f"  Doubts/Game:      {doubts_per_game:.2f}")
        print(f"  Proofs/Game:      {proofs_per_game:.2f}")
        print(f"  Swaps/Game:       {swaps_per_game:.2f}")
        print(f"  Physicat/Game:    {physicat_per_game:.2f}")

        print(f"  AvgClaimRank:     {avg_claim_rank:.2f}")
        print(f"  AvgClaimAmount:   {avg_claim_amount:.2f}")

        print(f"  GotDoubtedRate:   {got_doubted_rate:.3%}")
        print(f"  ClaimTruthRate:   {claim_truth_rate:.3%}")
        print(f"  FalseClaimRate:   {false_claim_rate:.3%}")
        print(f"  DoubtAccuracy:    {doubt_accuracy:.3%}")

        print(
            f"  ClaimTypes:       "
            f"A {alive_rate:.1%} / "
            f"D {dead_rate:.1%} / "
            f"E {empty_rate:.1%}"
        )

        print(f"  NoLegalActions:   {stats['no_legal_actions']}")
        print(f"  IllegalActions:   {stats['illegal_bot_actions']}")
        print(f"  MaxTurnReached:   {stats['max_turn_reached']}")

        if stats["tactical"]:
            print("  Tactical Analytics:")

            for key, value in stats["tactical"].items():
                print(f"    {key}: {value}")

        print()


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

            for seat in range(num_players):
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
        ("BayesianV4", BayesianV4Bot),
        ("TacticalBot", BayesianTacticalBot),
        ("TacticalV2", BayesianTacticalV2Bot),
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


def run_focused_tests(num_games_per_seat=500):
    evaluate_one_bot_vs_baseline(
        challenger_name="TacticalV2",
        challenger_factory=BayesianTacticalV2Bot,
        baseline_name="BayesianV1",
        baseline_factory=BayesianV1Bot,
        num_games_per_seat=num_games_per_seat,
        num_players=4,
    )

    evaluate_one_bot_vs_baseline(
        challenger_name="BayesianV1",
        challenger_factory=BayesianV1Bot,
        baseline_name="TacticalV2",
        baseline_factory=BayesianTacticalV2Bot,
        num_games_per_seat=num_games_per_seat,
        num_players=4,
    )

    evaluate_one_bot_vs_baseline(
        challenger_name="TacticalV2",
        challenger_factory=BayesianTacticalV2Bot,
        baseline_name="BayesianV4",
        baseline_factory=BayesianV4Bot,
        num_games_per_seat=num_games_per_seat,
        num_players=4,
    )

    evaluate_one_bot_vs_baseline(
        challenger_name="BayesianV4",
        challenger_factory=BayesianV4Bot,
        baseline_name="TacticalV2",
        baseline_factory=BayesianTacticalV2Bot,
        num_games_per_seat=num_games_per_seat,
        num_players=4,
    )


def run_xxl_evaluation():
    run_focused_tests(num_games_per_seat=1000)

    evaluate_bot_pool(
        num_games_per_rotation=1000,
    )


if __name__ == "__main__":
    # Fast test:
    # run_focused_tests(num_games_per_seat=200)

    # Medium test:
    run_focused_tests(num_games_per_seat=500)

    # XXL test:
    # run_xxl_evaluation()