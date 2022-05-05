"""TODO
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import *


def print_config(config_dict: dict, debug: bool = False) -> None:
    """Print config dictionary.If name of a key ends with 
    _debug then it is only printed if debug flag is set

    Parameters
    ----------
        config_dict: dict
            Config Dictionary containing data directories and other 
            experiment configs. If name of a key ends with _debug
            then it is only printed if debug flag is set
        debug: bool
            Flag for is debug mode
    """

    for key in config_dict:
        if key.split("_")[-1].lower() == "debug":
            if debug:
                print(f"{key} : {config_dict[key]}")
        else:
            print(f"{key} : {config_dict[key]}")


def class_distribution(root_path : str, classes : List[str]) -> dict:
    """TODO
    """

    distrib = {}
    for cls in classes:
        class_path = os.path.join(Path(root_path),Path(cls))
        distrib[cls] = len(os.listdir(class_path))
    return distrib

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

def sample_imgs(root_path : str, classes : list, sample_classes : float = 0.05, imgs_per_class: int =1)->dict:
    """TODO
    """
    
    sampled_imgs = {}
    classes = list(np.random.choice(classes,int(np.ceil(sample_classes*len(classes)))))
    for cls in classes:
        class_path = os.path.join(Path(root_path),Path(cls))
        rel_img_paths = list(np.random.choice(os.listdir(class_path),imgs_per_class))
        img_paths = []
        for rel_img_path in rel_img_paths:
            img_paths.append(os.path.join(Path(class_path),Path(rel_img_path)))
        sampled_imgs[cls] = img_paths
    return sampled_imgs


            