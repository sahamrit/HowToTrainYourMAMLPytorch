"""TODO
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import *


def plot_images_grid(columns: int , rows: int, images: List[str], figsize:Tuple[int,int] = (8,8))->None:
    """TODO
    """
    
    fig = plt.figure(figsize=figsize)
    columns = columns
    rows = rows
    assert rows*columns >= len(images), "Rows*Columns !>= len(images)"
    for i, img in enumerate(images):
        if i+1> len(images):
            break
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(plt.imread(img))
    plt.show()