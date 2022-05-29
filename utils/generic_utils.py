"""TODO
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import *

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


            