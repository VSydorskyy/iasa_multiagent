import math

from typing import Tuple

import numpy as np

from .base_model import _BaseModel

class DetermenisticChaosModel(_BaseModel):
    
    def __init__(
        self,
        n_points: int,
        field_size: Tuple[int, int],
        step_size: int,
        r: float,
        keep_trajoctories: bool = False
    ):
        super().__init__(
            n_points=n_points,
            field_size=field_size,
            step_size=step_size,
            keep_trajoctories=keep_trajoctories
        )
        
        self.r = r
        
        self.points = []
        self.angles = []
        self.real_angles = []
        
    def create_field(self):
        point_coords = [np.random.randint(0, f_size, self.n_points) for f_size in self.field_size]
        point_coords = np.stack(point_coords, axis=-1).astype(float) 
        
        angle = np.random.uniform(0, 1, self.n_points)
        
        self.angles.append(angle)
        self.real_angles.append(angle * 360)
        self.points.append(point_coords)
        self.markup_field(point_coords)
        
    def step(self):        
        current_coord = self.points[-1].copy()
        current_angle = self.angles[-1].copy()
        current_real_angle = self.real_angles[-1].copy()
        for i in range(current_coord.shape[0]):
            new_coord, new_angle, new_real_angle = self.step_function(current_coord[i], current_angle[i], current_real_angle[i])
            new_coord = self.continious_boarder_mode(new_coord)
            current_coord[i] = new_coord
            current_angle[i] = new_angle
            current_real_angle[i] = new_real_angle
            
        self.real_angles.append(current_real_angle)
        self.angles.append(current_angle)
        self.points.append(current_coord)
        self.markup_field(current_coord)
        
    def step_function(self, previous_coord: np.ndarray, angle: float, real_angle: float):
                
        new_angle = self.r * angle * (1 - angle)
        real_angle = (real_angle + (new_angle * 360)) % 360
        
        rad = math.radians(real_angle)
                
        previous_coord[0] += (math.cos(rad) * self.step_size)
        previous_coord[1] += (math.sin(rad) * self.step_size)
        
        previous_coord = self.continious_boarder_mode(previous_coord)
        
        return previous_coord, new_angle, real_angle
    
    def reset_partial(self):
        self.angles = []
        self.real_angles = []
        self.points = []