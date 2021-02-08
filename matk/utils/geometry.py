import math

import numpy as np


def angle_between_points(pointa: np.ndarray, pointb: np.ndarray):
    dx = pointb[0] - pointa[0]
    dy = pointb[1] - pointa[1]
    return math.degrees(math.atan2(dy, dx))
