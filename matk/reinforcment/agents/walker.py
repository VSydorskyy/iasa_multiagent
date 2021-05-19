from typing import Tuple

import numpy as np

from ...utils import exclude_list_from_list


class Walker(object):
    def __init__(self, coord: Tuple[int, int], main_step_prob: float = 1.0):
        self.start_coord = coord
        self.main_step_prob = main_step_prob
        self.coord = list(coord)

    def random_step(self, real_step, possible_actions):
        if len(possible_actions) == 1:
            return real_step

        if np.random.binomial(n=2, p=self.main_step_prob):
            return real_step
        else:
            return np.random.choice(
                exclude_list_from_list(possible_actions, [real_step])
            )

    def get_coord_from_action(self, action, cur_coord):
        x, y = cur_coord
        if action == "left":
            y -= 1
        elif action == "right":
            y += 1
        elif action == "up":
            x -= 1
        elif action == "down":
            x += 1
        return x, y

    def step(self, direction, possible_actions, do_step=True):
        if direction not in possible_actions:
            raise RuntimeError("direction should be in possible_actions")

        direction = self.random_step(direction, possible_actions)
        new_x, new_y = self.get_coord_from_action(direction, tuple(self.coord))
        self.coord[0] = new_x
        self.coord[1] = new_y

    def reset_coord(self):
        self.coord = list(self.start_coord)
