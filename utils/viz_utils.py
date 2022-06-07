"""This script contains fuctionalities for visualisation utilities

Functions it contains are listed below:
    *plot_images_grid: Plot images in a grid
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import *


def plot_images_grid(columns: int , rows: int, images: List[str], figsize:Tuple[int,int] = (8,8))->None:
    """Plot images in a grid

    Raises an exception if size of grid cant accomodate number of images

    Parameters
    ----------
    columns: int
        No of columns in the grid
    rows: int
        No of rows in the grid
    images: List[str]
        List of image paths 
    figsize: Tuple[int,int]
        Size of each figure
    
    Returns
    -------
    None
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