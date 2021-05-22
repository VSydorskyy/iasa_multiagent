import numpy as np

from axelrod.action import Action
from axelrod.player import Player

C, D = Action.C, Action.D


class GrimTrigger(Player):
    # These are various properties for the strategy
    name = "Grim Trigger"
    classifier = {
        "memory_depth": 1,
        "stochastic": False,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        super().__init__()
        self.trigger = False

    def strategy(self, opponent: Player) -> Action:
        """This is the actual strategy"""
        # First move
        if not self.history:
            return C

        self.trigger = opponent.history[-1] == D
        if self.trigger:
            return D
        else:
            return C
