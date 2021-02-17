import math

from typing import Tuple

import numpy as np

from .base_model import _BaseModel
from ..utils.geometry import angle_between_points


class CenterModel(_BaseModel):
    def __init__(
        self,
        n_points: int,
        field_size: Tuple[int, int],
        step_size: int,
        turn_degree: float,
        centre_accept_radius: float,
        point_accept_radius: float,
        more_point_to_accept: int = 1,
        keep_trajoctories: bool = False,
    ):
        super().__init__(
            n_points=n_points,
            field_size=field_size,
            step_size=step_size,
            keep_trajoctories=keep_trajoctories,
        )
        self.turn_degree = turn_degree
        self.centre_accept_radius = centre_accept_radius
        self.point_accept_radius = point_accept_radius
        self.more_point_to_accept = more_point_to_accept

        self.centre_point = np.array(
            [self.field_size[0] / 2, self.field_size[1] / 2]
        )

        self.points = []
        self.is_placed = []
        self.point_turn = None

    def create_field(self):
        point_coords = [
            np.random.randint(0, f_size, self.n_points)
            for f_size in self.field_size
        ]
        point_coords = np.stack(point_coords, axis=-1).astype(float)

        self.point_turn = np.random.choice(
            [-self.turn_degree, self.turn_degree], size=self.n_points
        )

        self.points.append(point_coords)
        self.is_placed.append(np.zeros(self.n_points).astype(bool))
        self.markup_field(point_coords, self.is_placed[-1].copy())

    def markup_field(self, points: np.ndarray, is_placed: np.ndarray):
        if self.keep_trajoctories and self.__len__() > 0:
            field = self.field_history[-1].copy()
        else:
            field = np.zeros(
                (self.field_size[0], self.field_size[1], 3), dtype=np.uint8
            )

        for is_p, coord in zip(is_placed, points):
            x_c, y_c = self.process_point_for_painting(coord)
            if is_p:
                field[x_c, y_c, 0] = 255
            else:
                field[x_c, y_c, 1] = 255

        self.field_history.append(field)

    def step(self):
        current_coord = self.points[-1].copy()
        current_is_placed = self.is_placed[-1].copy()

        for i in range(current_coord.shape[0]):
            new_coord, new_is_placed = self.step_function(
                current_coord[i], current_is_placed[i], self.point_turn[i]
            )
            new_coord = self.continious_boarder_mode(new_coord)
            current_is_placed[i] = new_is_placed
            current_coord[i] = new_coord

        self.stop = (~current_is_placed).sum() == 0
        self.is_placed.append(current_is_placed)
        self.points.append(current_coord)
        self.markup_field(current_coord, current_is_placed)

    def step_function(
        self, previous_coord: np.ndarray, is_placed: bool, turn_angle: float
    ):

        if not is_placed:

            angle_to_centre = angle_between_points(
                previous_coord, self.centre_point
            )

            angle = angle_to_centre + turn_angle

            rad = math.radians(angle)

            previous_coord[0] += math.cos(rad) * self.step_size
            previous_coord[1] += math.sin(rad) * self.step_size

            previous_coord = self.continious_boarder_mode(previous_coord)

            placed_points = self.points[-1][self.is_placed[-1]]

            dist_to_placed_points = np.linalg.norm(
                placed_points - previous_coord[None, :], axis=-1
            )

            point_diff_rule = (
                sum(dist_to_placed_points < self.point_accept_radius)
                > self.more_point_to_accept
            )
            centre_diff_rule = (
                np.linalg.norm(previous_coord - self.centre_point)
                < self.centre_accept_radius
            )

            is_placed = point_diff_rule or centre_diff_rule

        return previous_coord, is_placed

    def reset_partial(self):
        self.is_placed = []
        self.points = []
        self.point_turn = None
        self.stop = False
