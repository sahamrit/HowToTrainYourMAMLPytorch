"""TODO
"""

import os
from pathlib import Path
from typing import *

def imagenet_style_dataset_to_XY(data_dir: Path)-> Tuple[List[List[str]], List[str]]:
    """ Converts ImageNet style data directory structure
        to X,Y pairs.

    Parameters
    ----------

    data_dir: Path
        path of the data directory which contains data in 
        ImageNet style
    
    Returns
    -------
    Tuple[List[List[str]], List[str]]:
        (X,Y) where X is List of lists of all image file
        names per class and Y is a List of classes

    Additonal Information
    ---------------------
    ImageNet Directory Structure:
        root_dir
        |--train
        |  |--cls1
        |  |  |--a.png
        |  |--cls2
        |  |  |--x.png
        |--test
        |  |--cls3
        |  |  |--b.png
        |  |--cls4
        |  |  |--y.png
        |--val
        |  |--cls5
        |  |  |--c.png
        |  |--cls6
        |  |  |--z.png        
    """
    classes = set(os.listdir(data_dir))
    X = []; Y = []

    #TODO Replace by logger
    print(f"No of classes in {data_dir} are {len(classes)}")\

    for cls in classes:
        cls_path = os.path.join(data_dir,cls)
        Y.append(cls)
        X.append(list(os.listdir(cls_path)))

    return X,Y
