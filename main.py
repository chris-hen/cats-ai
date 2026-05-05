from training.self_play import play_one_game
from bots.random_bot import RandomBot
from bots.bayesian_bot import BayesianHeuristicBot


def main():
    print("RandomBot Test")
    result = play_one_game(
        num_players=4,
        bot_factory=RandomBot,
        verbose=False,
    )

    print("Winner:", result["winner"])
    print("Turns:", result["turns"])
    print()

    print("BayesianHeuristicBot Test")
    result = play_one_game(
        num_players=4,
        bot_factory=BayesianHeuristicBot,
        verbose=True,
    )

    print("Winner:", result["winner"])
    print("Turns:", result["turns"])


if __name__ == "__main__":
    main()