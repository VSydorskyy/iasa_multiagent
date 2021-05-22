import numpy as np

from axelrod.action import Action
from axelrod.player import Player

C, D = Action.C, Action.D


class RandomTitFor2Tats(Player):
    # These are various properties for the strategy
    name = "Random Tit For 2 Tats"
    classifier = {
        "memory_depth": 2,
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, p: float = 0.5, p_c: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.p_c = p_c

    def strategy(self, opponent: Player) -> Action:
        """This is the actual strategy"""
        # First move
        if not self.history:
            return C

        if np.random.binomial(n=2, p=self.p):
            return self._random.random_choice(self.p_c)
        else:
            return D if opponent.history[-2:] == [D, D] else C
