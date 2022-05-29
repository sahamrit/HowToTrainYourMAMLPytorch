"""TODO
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pprint

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
    print("Config Dictionary \n")
    for key in sorted(config_dict):
        if key.split("_")[-1].lower() == "debug" and not debug:
            continue
        if type(config_dict[key]) == dict:
            print(f"{key} => ")
            pprint.pprint(config_dict[key],width=1)
        else:
            print(f"{key} => {config_dict[key]}")


def class_distribution(root_path : str, classes : List[str]) -> dict:
    """TODO
    """

    distrib = {}
    for cls in classes:
        class_path = os.path.join(Path(root_path),Path(cls))
        distrib[cls] = len(os.listdir(class_path))
    return distrib

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


            