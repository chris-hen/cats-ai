import random

from cats.rules import legal_actions
from cats.actions import DoubtAction


class RandomBot:
    def choose_action(self, state):
        actions = legal_actions(state)

        if not actions:
            return DoubtAction()

        return random.choice(actions)