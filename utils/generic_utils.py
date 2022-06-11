"""This script contains generic utility functions

The functions it contains are:
    *class_distribution: This can give a distribution of images per class
    *sample_imgs: This can sample random classes and images per classes.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from pathlib import Path
from typing import *

def class_distribution(root_path : str, classes : List[str]) -> Dict[str,int]:
    """Gives a distribution of no of images per class.

    Parameters
    ----------
    root_path : str
        path which contains folders, where each folder represents a class
        and contains image of that class 
    classes: List[str]
        List of classes whose distribution is intended

    Returns
    -------
    Dict[str,int]:
        Dictionary with key as class names and value as number of images of 
        that class
    """

    distrib = {}
    for cls in classes:
        class_path = os.path.join(Path(root_path),Path(cls))
        distrib[cls] = len(os.listdir(class_path))
    return distrib

def sample_imgs(root_path : str, classes : list, sample_classes : float = 0.05, imgs_per_class: int =1) -> Dict[str,List[Path]]:
    """Samples classes at random and selects particular number of images per classes

    Parameters
    ----------
    root_path : str
        path which contains folders, where each folder represents a class
        and contains image of that class
    classes: List[str]
        List of candidate classes to sample from
    sample_classes: float  
        Fraction of classes to sample
    imgs_per_class: int
        Number of images per class to sample
    
    Returns
    ------
    Dict[str,List[Path]]
        A Dictionary of sampled images per class with keys 
        as sampled classes and value as list of image paths per class
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


#TODO desctiption of below

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    """TODO"""
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def load_param_dict(model: nn.Module, param_dict: dict) -> None:
    """
    Modifies params of the model explicitly which is different from
    load state dict. Load state dict method 
    of torch modifies inplace with no grad enabled 
    [with torch.no_grad(): param.copy_(input_param)]

    Parameter
    ---------
    model: nn.Module
        Any torch model inheriting from nn.Module
    param_dict: dict
        State dictionary of model containing new params
    
    Returns
    -------
    None
    """

    for name, param in param_dict.items():
        del_attr(model,name.split("."))
        set_attr(model,name.split("."),param)

def yeild_params(named_params:dict) -> List: 
    params = []
    for name, p in named_params.items():
        params.append(p)
    return params
        