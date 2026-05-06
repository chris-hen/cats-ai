from training.self_play import play_one_game
from bots.random_bot import RandomBot
from bots.bayesian_v1 import BayesianV1Bot
from bots.bayesian_v2 import BayesianV2Bot
from bots.bayesian_v3 import BayesianV3Bot
from bots.bayesian_v4 import BayesianV4Bot
from bots.bayesian_opponent import BayesianOpponentBot

import sys

from cats.play_cli import play_cli


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "play":
        play_cli()
    else:
        print("Usage:")
        print("  python main.py play")