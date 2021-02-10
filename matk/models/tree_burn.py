import math

from typing import Tuple, List

import numpy as np

from .base_model import _BaseModel

# R G B
TREE = [0, 255, 0]
BURNT = [100, 0, 0]
IN_FIRE = [255, 0, 0]
# Actions
T = 0
B = 1
F = 2


class TreeBurnModel(_BaseModel):
    def __init__(self, field_size: Tuple[int, int], forest_density: float):
        if forest_density > 1 or forest_density < 0:
            raise ValueError("forest_density should be in [0,1] interval")

        super().__init__(
            n_points=None,
            field_size=field_size,
            step_size=None,
            keep_trajoctories=False,
        )

        self.forest_density = forest_density
        self.action_history = []

    def create_field(self):
        tree_mask = np.random.binomial(
            n=1, p=self.forest_density, size=self.field_size
        ).astype(bool)
        action = np.zeros((*self.field_size, 4), dtype=bool)
        # Set trees
        action[:, :, T] = tree_mask
        # Burn trees
        action[action[:, 0, T], 0, F] = True
        action[action[:, 0, T], 0, T] = False

        self.action_history.append(action)
        self.field_history.append(self.convert_action2field(action))

    def step(self):
        action = self.action_history[-1].copy()
        x_coords, y_coords = np.where(action[:, :, F])
        for x, y in zip(x_coords, y_coords):
            action[x, y, F] = False
            action[x, y, B] = True
            for dx, dy in [(1, 0), (-1, 0), (0, 1)]:
                if (
                    (x + dx >= self.field_size[0])
                    or (y + dy >= self.field_size[1])
                    or (y + dy < 0)
                    or (x + dx < 0)
                ):
                    continue
                if action[x + dx, y + dy, T]:
                    action[x + dx, y + dy, T] = False
                    action[x + dx, y + dy, F] = True

        if action[:, :, F].sum() == 0:
            self.stop = True

        self.field_history.append(self.convert_action2field(action))
        self.action_history.append(action)

    def convert_action2field(self, action: np.ndarray):
        field = np.zeros((*self.field_size, 3), dtype=np.uint8)
        field[action[:, :, T]] = TREE
        field[action[:, :, F]] = IN_FIRE
        field[action[:, :, B]] = BURNT
        return field

    def reset_partial(self):
        self.action_history = []
