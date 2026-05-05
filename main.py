from training.self_play import play_one_game
from bots.random_bot import RandomBot
from bots.bayesian_v1 import BayesianV1Bot
from bots.bayesian_v2 import BayesianV2Bot
from bots.bayesian_v3 import BayesianV3Bot
from bots.bayesian_opponent import BayesianOpponentBot

def run_test(name, bot):
    print(f"\n=== {name} ===")

    result = play_one_game(
        num_players=4,
        bot_factory=bot,
        verbose=True,
    )

    print("Winner:", result["winner"])
    print("Turns:", result["turns"])


def main():
    run_test("RandomBot", RandomBot)
    run_test("BayesianV1Bot", BayesianV1Bot)
    run_test("BayesianV2Bot", BayesianV2Bot)
    run_test("BayesianV3Bot", BayesianV3Bot)
    run_test("OpponentBot", BayesianOpponentBot)


if __name__ == "__main__":
    main()