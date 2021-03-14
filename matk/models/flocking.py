from typing import Tuple

import numpy as np

from .base_model import _BaseModel


class FlockingModel(_BaseModel):
    def __init__(
        self,
        n_points: int,
        field_size: Tuple[int, int],
        step_size: int,
        cohere_speed: float = 0.125 * (3 / 5),
        align_speed: float = 0.125,
        separate_speed: float = 0.125 * (1.5 / 5),
        vision: float = 10,
        minimum_separation: float = 2,
    ):
        super().__init__(
            n_points=n_points,
            field_size=field_size,
            step_size=step_size,
            keep_trajoctories=False,
        )

        self.points = []
        self.cohere_speed = cohere_speed
        self.align_speed = align_speed
        self.separate_speed = separate_speed

        self.vision = vision
        self.minimum_separation = minimum_separation

    def create_field(self):
        point_coords = [
            np.random.randint(0, f_size, self.n_points)
            for f_size in self.field_size
        ]
        # Add velocities
        point_coords += [
            np.random.uniform(low=-10, high=10, size=self.n_points)
            for _ in self.field_size
        ]
        # Stack
        point_coords = np.stack(point_coords, axis=-1).astype(float)
        # Normalize velocity
        point_coords[:, 2:] = (
            point_coords[:, 2:]
            / np.linalg.norm(point_coords[:, 2:], axis=-1)[:, None]
        )

        self.points.append(point_coords)
        self.markup_field(point_coords)

    def move_by_velocity(self, coords, velocities):
        steped_coord = coords + velocities * self.step_size
        for i in range(steped_coord.shape[0]):
            steped_coord[i] = self.continious_boarder_mode(steped_coord[i])
        return steped_coord

    # vector from coord to opposite
    def get_torus_diff(self, coord, oposite_coord):
        diff = oposite_coord - coord

        w, h = self.field_size[0], self.field_size[1]

        if np.abs(diff[0]) > w / 2:
            diff[0] -= w * np.sign(diff[0])
        if np.abs(diff[1]) > h / 2:
            diff[1] -= h * np.sign(diff[1])
        return diff

    def get_torus_distances(self, coord, coords):
        d = np.abs(coords - coord[np.newaxis, ...])
        dx = d[:, 0]
        dy = d[:, 1]

        tour_dx = np.minimum(dx, self.field_size[0] - dx)
        tour_dy = np.minimum(dy, self.field_size[1] - dy)

        dist = np.sqrt(tour_dx ** 2 + tour_dy ** 2)
        return dist

    def cohere_velocity(self, velocity, coord, coords):
        direction = self.get_torus_diff(coord, np.mean(coords, 0))
        if np.sum(direction) > 0:
            direction /= np.linalg.norm(direction)
        return velocity + direction * self.cohere_speed

    def align_velocity(self, velocity, velocities):
        mean_direction = np.mean(velocities, 0)
        direction = mean_direction - velocity
        if np.sum(direction) > 0:
            direction /= np.linalg.norm(direction)
        return velocity + direction * self.align_speed

    def separate_velocity(self, velocity, coord, oposite_coord):
        direction = -self.get_torus_diff(coord, oposite_coord)
        if np.sum(direction) > 0:
            direction /= np.linalg.norm(direction)
        return velocity + direction * self.separate_speed

    def do_flock(self, coord, velocity, bird_id, coords, velocities):
        other_coords = np.delete(coords, bird_id, 0)
        other_velocities = np.delete(velocities, bird_id, 0)

        distances = self.get_torus_distances(coord, other_coords)

        flockmates = other_coords[distances < self.vision]
        if len(flockmates) > 0:
            if np.min(distances) < self.minimum_separation:
                neighbour = other_coords[np.argmin(distances)]
                velocity = self.separate_velocity(velocity, coord, neighbour)
                velocity /= np.linalg.norm(velocity)
            else:
                flockmates_velocities = other_velocities[
                    distances < self.vision
                ]
                velocity = self.align_velocity(velocity, flockmates_velocities)
                velocity /= np.linalg.norm(velocity)
                velocity = self.cohere_velocity(velocity, coord, flockmates)
                velocity /= np.linalg.norm(velocity)

        return velocity

    # need to add all find_neighbours and done
    def step(self):
        # x, y, vel_x, vel_y
        current_coords = self.points[-1].copy()
        new_coords = np.zeros_like(current_coords)

        for bird_id, bird in enumerate(current_coords):
            bird_coord = bird[:2]
            bird_velocity = bird[2:]
            new_bird_velocity = self.do_flock(
                bird_coord,
                bird_velocity,
                bird_id,
                current_coords[:, :2],
                current_coords[:, 2:],
            )
            new_coords[bird_id, :2] = bird_coord
            new_coords[bird_id, 2:] = new_bird_velocity

        # Move
        new_coords[:, :2] = self.move_by_velocity(
            new_coords[:, :2], new_coords[:, 2:4]
        )

        self.points.append(new_coords)
        self.markup_field(new_coords)

    def reset_partial(self):
        self.points = []
        self.stop = False
