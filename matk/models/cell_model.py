import math

from typing import Tuple, List

import numpy as np

from matk.models.base_model import _BaseModel

# R G B
ALIVE = [0, 255, 0]
NOT_REPL = [255, 0, 0]
REPL = [0, 0, 255]
# Model types
MODEL_TYPES = ["top", "random_top", "center", "random_center"]
# Actions
R = 0
A = 1
NR = 2


class CellModel(_BaseModel):
    def __init__(
        self,
        field_size: Tuple[int, int],
        model_type: str,
        n_directions: int = 2,
    ):
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type should be one of {MODEL_TYPES}")

        super().__init__(
            n_points=None,
            field_size=field_size,
            step_size=None,
            keep_trajoctories=False,
        )

        self.model_type = model_type
        self.n_directions = n_directions
        self.action_history = []

    def create_field(self):
        birth = np.zeros((*self.field_size, 3), dtype=bool)
        # Set first cell
        if self.model_type in ["top", "random_top"]:
            birth[0, self.field_size[1] // 2, A] = True
            birth[0, self.field_size[1] // 2, R] = True
        elif self.model_type in ["center", "random_center"]:
            birth[self.field_size[0] // 2, self.field_size[1] // 2, A] = True
            birth[self.field_size[0] // 2, self.field_size[1] // 2, R] = True

        self.action_history.append(birth)
        self.field_history.append(self.convert_action2field(birth))

    def _generate_childs(self, x: int, y: int):
        if self.model_type == "top":
            return [(x + 1, y + 1), (x + 1, y - 1)]
        elif self.model_type == "random_top":
            if np.random.binomial(n=1, p=0.5):
                return [(x + 1, y + 1)]
            else:
                return [(x + 1, y - 1)]
        elif self.model_type == "center":
            return [
                (x + 1, y + 1),
                (x + 1, y - 1),
                (x - 1, y + 1),
                (x - 1, y - 1),
            ]
        elif self.model_type == "random_center":
            indices = np.random.choice(
                [0, 1, 2, 3], size=self.n_directions, replace=False
            )
            directions = [
                (x + 1, y + 1),
                (x + 1, y - 1),
                (x - 1, y + 1),
                (x - 1, y - 1),
            ]
            return [directions[i] for i in indices]

    def step(self):
        birth = self.action_history[-1].copy()
        x_coords, y_coords = np.where(birth[:, :, R])
        for x, y in zip(x_coords, y_coords):

            childs = self._generate_childs(x, y)

            # In field
            in_fields = []
            for child in childs:
                cur_in_field = True
                for x_i in [0, 1]:
                    cur_in_field = cur_in_field and (
                        0 <= child[x_i] < self.field_size[x_i]
                    )
                in_fields.append(cur_in_field)

            # Make new generation
            for i in range(len(childs)):
                child = childs[i]
                if (
                    in_fields[i]
                    and (not birth[child][A])
                    and (not birth[child][NR])
                ):
                    birth[child][A] = True
                    birth[child][R] = True
                elif (
                    in_fields[i] and birth[child][A] and (not birth[child][NR])
                ):
                    birth[child][A] = False
                    birth[child][R] = False
                    birth[child][NR] = True

            # Kill old one
            birth[x, y, R] = False

        # No reproductive cells
        self.stop = birth[:, :, R].sum() == 0

        self.field_history.append(self.convert_action2field(birth))
        self.action_history.append(birth)

    def convert_action2field(self, action: np.ndarray):
        field = np.zeros((*self.field_size, 3), dtype=np.uint8)
        field[action[:, :, A]] = ALIVE
        field[action[:, :, R]] = REPL
        field[action[:, :, NR]] = NOT_REPL
        return field

    def reset_partial(self):
        self.action_history = []
        self.stop = False
