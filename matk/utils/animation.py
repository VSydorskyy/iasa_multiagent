from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation

def animate_frames(
    frame_list: List[np.ndarray],
    figsize: Tuple[int, int] = (5, 5),
    interval: int = 10,
    repeat_delay: int = 10
):
    fig = plt.figure(figsize=figsize)
    ims = []

    for i in range(len(frame_list)):
        im = plt.imshow(frame_list[i], animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(
        fig, 
        ims, 
        interval=interval, 
        blit=True, 
        repeat_delay=repeat_delay
    )

    return ani