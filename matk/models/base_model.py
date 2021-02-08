from typing import Tuple

import numpy as np

class _BaseModel(object):
    
    def __init__(
        self,
        n_points: int,
        field_size: Tuple[int, int],
        step_size: int,
        keep_trajoctories: bool = False
    ):
        self.field_size = field_size
        self.n_points = n_points
        self.step_size = step_size
        self.keep_trajoctories = keep_trajoctories
                
        self.field_history = []
        
    def create_field(self):
        raise NotImplementedError("create_field not implemented")
        
    def markup_field(self, points:np.ndarray):
        if self.keep_trajoctories and self.__len__() > 0:
            field = self.field_history[-1].copy()
        else:
            field = np.zeros(self.field_size)
            
        for coord in points:
            x_c, y_c = self.process_point_for_painting(coord)
            field[x_c,y_c] = 1
            
        self.field_history.append(field)
        
    def process_point_for_painting(self, point_coords: np.ndarray):
        coords = [int(round(point_coords[0])), int(round(point_coords[1]))]
        for x_y  in [0, 1]:
            if coords[x_y] < 0:
                coords[x_y] = self.field_size[x_y] - 1
            elif coords[x_y] >= self.field_size[x_y]:
                coords[x_y] = 0
        return coords[0], coords[1]
    
    def continious_boarder_mode(self, point_coords: np.ndarray):
        for x_y in [0, 1]:
            if point_coords[x_y] > 0:
                point_coords[x_y] = point_coords[x_y] % self.field_size[x_y]
            else:
                point_coords[x_y] = self.field_size[x_y] + point_coords[x_y]
        return point_coords
        
    def step(self):        
        raise NotImplementedError("step not implemented")
        
    def __len__(self):
        return len(self.field_history)
    
    def __getitem__(self, idx: int):
        return self.field_history[idx]
    
    def reset_partial(self):
        raise NotImplementedError("reset_partial not implemented")

    def reset(self):
        self.field_history = []
        self.reset_partial()
        
    def run_n_steps(self, n: int):
        self.reset()
        self.create_field()
        for _ in range(n):
            self.step()
            
    def run_more_n_steps(self, n: int):
        for _ in range(n):
            self.step()