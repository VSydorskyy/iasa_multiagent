import math

from typing import Tuple, List

import numpy as np

from .base_model import _BaseModel

# R G B
TREE = [0, 255, 0]
BURNT = [100, 0, 0]
IN_FIRE = [255, 0, 0]
LAKE = [0, 0, 255]
# Actions
T = 0
B = 1
F = 2
L = 3


def check_interval(variable, variable_name, min_v, max_v):
    if variable > max_v or variable < min_v:
        raise ValueError(
            f"{variable_name} should be in [{min_v},{max_v}] interval"
        )


class TreeBurnModel(_BaseModel):
    def __init__(
        self,
        field_size: Tuple[int, int],
        forest_density: float,
        activate_wind: bool = False,
        #         wind_angle: float = 0
        horizontal_wind: float = 0.0,
        vertical_wind: float = 0.0,
        n_lakes: int = 0,
        lake_area: int = 0,
    ):
        check_interval(forest_density, "forest_density", 0, 1)
        check_interval(horizontal_wind, "horizontal_wind", -25, 25)
        check_interval(vertical_wind, "vertical_wind", -25, 25)
        check_interval(n_lakes, "n_lakes", 0, 5)
        check_interval(
            lake_area, "lake_area", 0, min(field_size[0], field_size[1]) * 0.5
        )
        #         check_interval(wind_angle, 'wind_angle', 0, 360)

        super().__init__(
            n_points=None,
            field_size=field_size,
            step_size=None,
            keep_trajoctories=False,
        )

        self.activate_wind = activate_wind
        if horizontal_wind == 0:
            self.right_p = None
        else:
            self.right_p = (horizontal_wind + 25) / 50
        if vertical_wind == 0:
            self.up_p = None
        else:
            self.up_p = (vertical_wind + 25) / 50
        #         wind_rad = math.radians(wind_angle)
        #         self.right_p = 1 - ((math.cos(wind_rad) + 1) / 2)
        #         self.up_p = (math.sin(wind_rad) + 1) / 2

        self.n_lakes = n_lakes
        self.lake_area = lake_area

        self.forest_density = forest_density
        self.action_history = []

    def place_lakes(self, action):
        x_lefts = np.random.randint(
            low=0, high=self.field_size[1] - self.lake_area, size=self.n_lakes
        )
        y_lefts = np.random.randint(
            low=0, high=self.field_size[0] - self.lake_area, size=self.n_lakes
        )

        for x, y in zip(x_lefts, y_lefts):
            action[x : x + self.lake_area, y : y + self.lake_area, T] = False
            action[x : x + self.lake_area, y : y + self.lake_area, L] = True

        return action

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
        # Place lakes
        if self.n_lakes > 0:
            action = self.place_lakes(action)

        self.action_history.append(action)
        self.field_history.append(self.convert_action2field(action))

    def burn_one(self, action, x_cor, y_cor):
        if (
            (x_cor >= self.field_size[0])
            or (y_cor >= self.field_size[1])
            or (y_cor < 0)
            or (x_cor < 0)
        ):
            return action, True
        if action[x_cor, y_cor, T]:
            action[x_cor, y_cor, T] = False
            action[x_cor, y_cor, F] = True
            return action, False

        return action, False

    def step(self):
        action = self.action_history[-1].copy()
        x_coords, y_coords = np.where(action[:, :, F])
        for x, y in zip(x_coords, y_coords):
            action[x, y, F] = False
            action[x, y, B] = True
            if self.activate_wind:

                if self.right_p is None:
                    right_step = True
                    left_step = False
                else:
                    right_step = bool(np.random.binomial(n=1, p=self.right_p))
                    left_step = not right_step

                if self.up_p is None:
                    up_step = True
                    down_step = True
                else:
                    up_step = bool(np.random.binomial(n=1, p=self.up_p))
                    down_step = not up_step

                for dx, dy, p in [
                    (1, 0, down_step),
                    (-1, 0, up_step),
                    (0, 1, right_step),
                    (0, -1, left_step),
                ]:
                    if p:
                        action, do_continue = self.burn_one(
                            action, x + dx, y + dy
                        )
                        if do_continue:
                            continue
            else:
                for dx, dy in [(1, 0), (-1, 0), (0, 1)]:
                    action, do_continue = self.burn_one(action, x + dx, y + dy)
                    if do_continue:
                        continue

        if action[:, :, F].sum() == 0:
            self.stop = True

        self.field_history.append(self.convert_action2field(action))
        self.action_history.append(action)

    def convert_action2field(self, action: np.ndarray):
        field = np.zeros((*self.field_size, 3), dtype=np.uint8)
        field[action[:, :, T]] = TREE
        field[action[:, :, F]] = IN_FIRE
        field[action[:, :, B]] = BURNT
        field[action[:, :, L]] = LAKE
        return field

    def reset_partial(self):
        self.action_history = []
