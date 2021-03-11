import math

from typing import Tuple, List, Optional

import numpy as np

from .base_model import _BaseModel

# R G B
TREE = [0, 255, 0]
SLOW_TREE = [0, 0, 155]
BURNT = [100, 0, 0]
IN_FIRE = [255, 0, 0]
LAKE = [0, 0, 255]
GRASS = [0, 100, 0]
# Actions
T = 0
B = 1
F = 2
L = 3
ST = 4
G = 5
# Other consts
SLOW_BURN = 5


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
        slow_tree_density: Optional[float] = None,
        slow_tree_burn_prob: float = 0.5,
        activate_wind: bool = False,
        horizontal_wind: float = 0.0,
        vertical_wind: float = 0.0,
        n_lakes: int = 0,
        lake_area: int = 0,
        n_grasses: int = 0,
        grass_area: int = 0,
    ):
        check_interval(forest_density, "forest_density", 0, 1)
        check_interval(horizontal_wind, "horizontal_wind", -25, 25)
        check_interval(vertical_wind, "vertical_wind", -25, 25)
        check_interval(n_lakes, "n_lakes", 0, 5)
        check_interval(
            lake_area, "lake_area", 0, min(field_size[0], field_size[1]) * 0.5
        )
        check_interval(n_grasses, "n_grasses", 0, 5)
        check_interval(
            grass_area,
            "grass_area",
            0,
            min(field_size[0], field_size[1]) * 0.5,
        )
        if slow_tree_density is not None:
            self.use_slow_tree = True
            check_interval(slow_tree_density, "slow_tree_density", 0, 1)
            check_interval(slow_tree_burn_prob, "slow_tree_burn_prob", 0, 1)
            self.slow_tree_burn_prob = slow_tree_burn_prob
            self.slow_tree_density = slow_tree_density
            self.burn_time = dict()
        else:
            self.use_slow_tree = False

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

        self.n_lakes = n_lakes
        self.lake_area = lake_area

        self.n_grasses = n_grasses
        self.grass_area = grass_area
        if n_grasses > 0:
            self.grasses = []

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
            action[x : x + self.lake_area, y : y + self.lake_area, :] = False
            action[x : x + self.lake_area, y : y + self.lake_area, L] = True

        return action

    def place_grass(self, action):
        x_lefts = np.random.randint(
            low=0,
            high=self.field_size[1] - self.grass_area,
            size=self.n_grasses,
        )
        y_lefts = np.random.randint(
            low=0,
            high=self.field_size[0] - self.grass_area,
            size=self.n_grasses,
        )

        for x, y in zip(x_lefts, y_lefts):
            action[x : x + self.grass_area, y : y + self.grass_area, :] = False
            action[x : x + self.grass_area, y : y + self.grass_area, G] = True
            self.grasses.append((x, y))

        return action

    def place_slow_trees(self, action):
        tree_placed = np.where(action[:, :, T])
        for x, y in zip(*tree_placed):
            if np.random.binomial(n=1, p=self.slow_tree_density):
                action[x, y, T] = False
                action[x, y, ST] = True
                self.burn_time[str(x) + str(y)] = 0
        return action

    def create_field(self):
        tree_mask = np.random.binomial(
            n=1, p=self.forest_density, size=self.field_size
        ).astype(bool)
        action = np.zeros((*self.field_size, 6), dtype=bool)
        # Set trees
        action[:, :, T] = tree_mask
        # Pace Grass
        if self.n_grasses > 0:
            action = self.place_grass(action)
        # Burn trees
        action[action[:, 0, T], 0, F] = True
        action[action[:, 0, T], 0, T] = False
        action[action[:, 0, T], 0, G] = False
        # Place lakes
        if self.n_lakes > 0:
            action = self.place_lakes(action)
        # Place slow trees
        if self.use_slow_tree:
            action = self.place_slow_trees(action)

        self.action_history.append(action)
        self.field_history.append(self.convert_action2field(action))

    def burn_grass(self, action, x_cor, y_cor):
        for x_g, y_g in self.grasses:
            if (x_g <= x_cor < x_g + self.grass_area) and (
                y_g <= y_cor < y_g + self.grass_area
            ):
                action[
                    x_g : x_g + self.grass_area, y_g : y_g + self.grass_area, F
                ] = True
                action[
                    x_g : x_g + self.grass_area,
                    y_cor : y_g + self.grass_area,
                    G,
                ] = False

        return action

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
        elif action[x_cor, y_cor, ST]:
            action[x_cor, y_cor, F] = True
            self.burn_time[str(x_cor) + str(y_cor)] += 1
            return action, False
        elif action[x_cor, y_cor, G]:
            action = self.burn_grass(action, x_cor, y_cor)
            return action, False

        return action, False

    def step(self):
        action = self.action_history[-1].copy()
        x_coords, y_coords = np.where(action[:, :, F])
        for x, y in zip(x_coords, y_coords):
            # Burn Slow tree
            if action[x, y, ST] and action[x, y, F]:
                self.burn_time[str(x) + str(y)] += 1
                if self.burn_time[str(x) + str(y)] >= 5:
                    if np.random.binomial(n=1, p=self.slow_tree_burn_prob):
                        action[x, y, ST] = False
                        action[x, y, F] = False
                        action[x, y, B] = True
                    else:
                        action[x, y, F] = False
            # Burn ordinary trees
            else:
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
        field[action[:, :, ST]] = SLOW_TREE
        field[action[:, :, F]] = IN_FIRE
        field[action[:, :, B]] = BURNT
        field[action[:, :, L]] = LAKE
        field[action[:, :, G]] = GRASS
        return field

    def reset_partial(self):
        self.action_history = []
        self.burn_time = dict()
