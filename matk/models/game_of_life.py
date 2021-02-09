import math

from typing import Tuple, List

import numpy as np

from .base_model import _BaseModel


class GameOfLifeModel(_BaseModel):
    def __init__(
        self, field_size: Tuple[int, int], percent_of_positive_points: float
    ):
        if percent_of_positive_points > 1 or percent_of_positive_points < 0:
            raise ValueError(
                "percent_of_positive_points should be in [0,1] interval"
            )

        super().__init__(
            n_points=None,
            field_size=field_size,
            step_size=None,
            keep_trajoctories=False,
        )

        self.percent_of_positive_points = percent_of_positive_points

    def create_field(self):
        field = np.random.binomial(
            n=1, p=self.percent_of_positive_points, size=self.field_size
        ).astype(int)
        self.field_history.append(field)

    def step(self):
        previous_field = self.field_history[-1].copy()
        new_field = np.zeros_like(previous_field, dtype=int)
        for i in range(previous_field.shape[0]):
            for j in range(previous_field.shape[1]):
                is_alive = self.step_function([i, j], previous_field[i, j])
                new_field[i, j] = is_alive

        self.field_history.append(new_field)

    def compute_neighbours(self, coord: np.ndarray):
        neighbors_sum = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                coords = self.process_point_for_painting(
                    [coord[0] + dx, coord[1] + dy]
                )
                neighbors_sum += self.field_history[-1][coords]

        return neighbors_sum

    def step_function(self, coord: List[int], was_alive: bool):
        n_neighbors = self.compute_neighbours(coord)

        if was_alive:
            if (n_neighbors < 2) or (n_neighbors > 3):
                return 0
            else:
                return 1
        else:
            if n_neighbors == 3:
                return 1
            else:
                return 0

    def reset_partial(self):
        pass
