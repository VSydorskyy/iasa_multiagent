from typing import Tuple, Optional

import numpy as np

from matplotlib import pyplot as plt

from ...utils import exclude_list_from_list
from ..agents.walker import Walker

# R G B
START = ((0, 255, 0), 1)
END = ((0, 0, 100), 2)
VOLCANO = ((255, 0, 0), 3)
ROCK = ((96, 96, 96), 4)
WALKER = ((255, 51, 153), 5)
ESCALATOR = ((0, 255, 255), 6)


class Labirint(object):
    def __init__(
        self,
        size: int,
        n_rocks: int,
        n_volcanos: int,
        n_escalators: int,
        escalator_len: int,
        main_step_prob: float = 1.0,
        start: Tuple[int, int] = (0, 0),
        end: Optional[Tuple[int, int]] = None,
    ):
        self.size = size
        self.n_rocks = n_rocks
        self.n_volcanos = n_volcanos
        self.n_escalators = n_escalators
        self.escalator_len = escalator_len
        self.start = start
        self.end = (size - 1, size - 1) if end is None else end

        field, states = self.create_field()
        self.field = field
        self.states = states
        self.agent = Walker(start, main_step_prob=main_step_prob)

    def create_field(self):
        field = np.zeros((self.size, self.size))
        all_possible_coords = [
            (i, j) for i in range(self.size) for j in range(self.size)
        ]

        # Place start/end
        field[self.start] = START[1]
        field[self.end] = END[1]
        all_possible_coords = exclude_list_from_list(
            all_possible_coords, [self.start, self.end]
        )

        # Place escalator
        ecalator_possible_coords = [
            el
            for el in all_possible_coords
            if (el[1] != self.size - 1)
            and (el[0] + self.escalator_len + 1 < self.size)
        ]
        escalator_places_idx = np.random.choice(
            list(range(len(ecalator_possible_coords))), size=self.n_escalators
        )
        escalator_places = []
        for i in escalator_places_idx:
            field[
                ecalator_possible_coords[i][0] : ecalator_possible_coords[i][0]
                + self.escalator_len,
                ecalator_possible_coords[i][1],
            ] = ESCALATOR[1]
            escalator_places += [
                (
                    ecalator_possible_coords[i][0] + j,
                    ecalator_possible_coords[i][1],
                )
                for j in range(self.escalator_len + 1)
            ]
        all_possible_coords = exclude_list_from_list(
            all_possible_coords, list(escalator_places)
        )

        # Place rocks
        rock_places_idx = np.random.choice(
            list(range(len(all_possible_coords))), size=self.n_rocks
        )
        rock_places = []
        for i in rock_places_idx:
            field[all_possible_coords[i]] = ROCK[1]
            rock_places.append(all_possible_coords[i])
        all_possible_coords = exclude_list_from_list(
            all_possible_coords, list(rock_places)
        )

        # Place volkanos
        volcano_places_idx = np.random.choice(
            list(range(len(all_possible_coords))), size=self.n_volcanos
        )
        volcano_places = []
        for i in volcano_places_idx:
            field[all_possible_coords[i]] = VOLCANO[1]
            volcano_places.append(all_possible_coords[i])
        all_possible_coords = exclude_list_from_list(
            all_possible_coords, list(volcano_places)
        )

        states = [(i, j) for i in range(self.size) for j in range(self.size)]
        states = exclude_list_from_list(states, list(rock_places))

        return field, states

    def v_print(self, input_field=None, with_agent=False, return_field=False):
        field = self.field if input_field is None else input_field

        visualise_field = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        # Visualise start/end
        visualise_field[self.start] = START[0]
        visualise_field[self.end] = END[0]

        # Visualise escalator
        rock_coords = np.where(field == ESCALATOR[1])
        for x, y in zip(rock_coords[0], rock_coords[1]):
            visualise_field[x, y] = ESCALATOR[0]

        # Visualise rocks
        rock_coords = np.where(field == ROCK[1])
        for x, y in zip(rock_coords[0], rock_coords[1]):
            visualise_field[x, y] = ROCK[0]

        # Visualise volcano
        volcano_coords = np.where(field == VOLCANO[1])
        for x, y in zip(volcano_coords[0], volcano_coords[1]):
            visualise_field[x, y] = VOLCANO[0]

        # Visualise walker
        if with_agent:
            visualise_field[self.agent.coord[0], self.agent.coord[1]] = WALKER[
                0
            ]

        if return_field:
            return visualise_field
        else:
            plt.imshow(visualise_field)
            plt.plot()

    def get_possible_actions(self, coords):
        if self.field[coords[0], coords[1]] == ESCALATOR[1]:
            return ["down"]

        possible_actions = []
        if (
            coords[1] - 1 >= 0
            and self.field[coords[0], coords[1] - 1] != ROCK[1]
        ):
            possible_actions.append("left")
        if (
            coords[1] + 1 < self.size
            and self.field[coords[0], coords[1] + 1] != ROCK[1]
        ):
            possible_actions.append("right")
        if (
            coords[0] - 1 >= 0
            and self.field[coords[0] - 1, coords[1]] != ROCK[1]
        ):
            possible_actions.append("up")
        if (
            coords[0] + 1 < self.size
            and self.field[coords[0] + 1, coords[1]] != ROCK[1]
        ):
            possible_actions.append("down")

        return possible_actions

    def get_reward(self, coords):
        if self.field[coords[0], coords[1]] in [0, START[1], ESCALATOR[1]]:
            return -1
        if self.field[coords[0], coords[1]] == VOLCANO[1]:
            return -50
        if self.field[coords[0], coords[1]] == END[1]:
            return 100

    def get_endgame(self, coords):
        if self.field[coords[0], coords[1]] in [VOLCANO[1], END[1]]:
            return True
        else:
            return False

    def get_success(self, coords):
        if self.field[coords[0], coords[1]] == END[1]:
            return True
        else:
            return False

    def get_escalator(self, coords):
        if self.field[coords[0], coords[1]] == ESCALATOR[1]:
            return True
        else:
            return False
