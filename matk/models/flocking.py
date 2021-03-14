from typing import Tuple, Optional

import numpy as np

from .base_model import _BaseModel

BIRD_INTENSITY = 0.5
EAGLE_INTENSITY = 1


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
        n_eagles: int = 0,
        eagles_speed: int = 0.125,
        eagles_step_size: int = 1,
        eagles_attack_radius: float = 3,
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

        self.use_eagles = n_eagles > 0
        if self.use_eagles:
            self.eagles = []
        self.n_eagles = n_eagles
        self.eagles_speed = eagles_speed
        self.eagles_step_size = eagles_step_size
        self.eagles_attack_radius = eagles_attack_radius

    def markup_field(self, points: np.ndarray, egales: Optional[np.ndarray]):
        if self.keep_trajoctories and self.__len__() > 0:
            field = self.field_history[-1].copy()
        else:
            field = np.zeros(self.field_size)

        for coord in points:
            x_c, y_c = self.process_point_for_painting(coord)
            field[x_c, y_c] = BIRD_INTENSITY

        if egales is not None:
            for coord in egales:
                x_c, y_c = self.process_point_for_painting(coord)
                field[x_c, y_c] = EAGLE_INTENSITY

        self.field_history.append(field)

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

        if self.n_eagles > 0:
            eagles_coords = [
                np.random.randint(0, f_size, self.n_eagles)
                for f_size in self.field_size
            ]
            # Add velocities
            eagles_coords += [
                np.random.uniform(low=-10, high=10, size=self.n_eagles)
                for _ in self.field_size
            ]
            # Stack
            eagles_coords = np.stack(eagles_coords, axis=-1).astype(float)
            # Normalize velocity
            eagles_coords[:, 2:] = (
                eagles_coords[:, 2:]
                / np.linalg.norm(eagles_coords[:, 2:], axis=-1)[:, None]
            )
            self.eagles.append(eagles_coords)

        self.points.append(point_coords)
        self.markup_field(
            point_coords, eagles_coords if self.use_eagles else None
        )

    def move_by_velocity(self, coords, velocities, step_size):
        steped_coord = coords + velocities * step_size
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

    def do_eagle(self, eagle_velocity, eagle_coord, birds):
        # import ipdb; ipdb.set_trace()
        dist_to_birds = self.get_torus_distances(eagle_coord, birds)

        nearest_bird = birds[np.argmin(dist_to_birds)]
        direction = self.get_torus_diff(eagle_coord, nearest_bird)
        if np.sum(direction) > 0:
            direction /= np.linalg.norm(direction)

        updated_velocity = eagle_velocity + direction * self.eagles_speed
        updated_velocity /= np.linalg.norm(updated_velocity)
        return updated_velocity

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

        # Move birds
        new_coords[:, :2] = self.move_by_velocity(
            new_coords[:, :2], new_coords[:, 2:], self.step_size
        )

        if self.use_eagles:
            current_eagles = self.eagles[-1]
            new_eagles = np.zeros_like(current_eagles)

            for eagle_id, eagle in enumerate(current_eagles):
                eagle_coord = eagle[:2]
                eagle_velocity = eagle[2:]
                new_eagle_velocity = self.do_eagle(
                    eagle_velocity, eagle_coord, new_coords[:, :2]
                )
                new_eagles[eagle_id, :2] = eagle_coord
                new_eagles[eagle_id, 2:] = new_eagle_velocity

            # Move eagles
            new_eagles[:, :2] = self.move_by_velocity(
                new_eagles[:, :2], new_eagles[:, 2:], self.eagles_step_size
            )
            self.eagles.append(new_eagles)

            # Kill birds
            killed_birds = []
            for bird_id, bird in enumerate(new_coords):
                dist_to_eagle = self.get_torus_distances(
                    bird[:2], new_eagles[:, :2]
                )
                if np.min(dist_to_eagle) <= self.eagles_attack_radius:
                    killed_birds.append(bird_id)
            new_coords = np.delete(new_coords, killed_birds, 0)

        self.points.append(new_coords)
        self.markup_field(new_coords, new_eagles if self.use_eagles else None)
        self.stop = len(new_coords) == 0

    def reset_partial(self):
        self.points = []
        self.stop = False
        if self.use_eagles:
            self.eagles = []
