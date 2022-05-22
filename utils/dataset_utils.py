"""TODO
"""

import imp
import os
import numpy as np
import copy

from pathlib import Path
from typing import *

def imagenet_style_dataset_to_XY(data_dir: Path)-> Tuple[List[List[str]], List[int], Dict[str,int]]:
    """ Converts ImageNet style data directory structure to X,Y pairs.

    Parameters
    ----------
    data_dir: Path
        path of the data directory which contains data in ImageNet style
    
    Returns
    -------
    Tuple[List[List[str]], List[int], Dict[str,int]]:
        (X,Y,cls_int_map) where X is List of lists of all image file names per class 
        , Y is a List of classes and cls_int_map is an integer map of string classes

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
    cls_int_map = {}
    for i , cls in enumerate(classes):
        cls_int_map[cls] = i

    X = []; Y = []

    #TODO Replace by logger
    print(f"No of classes in {data_dir} are {len(classes)}")\
        
    for cls in classes:
        cls_path = os.path.join(data_dir,cls)
        Y.append(cls_int_map[cls])
        imgs = list(os.listdir(cls_path))
        img_pth_list = []
        for img in imgs:
            img_pth_list.append(str(os.path.join(cls_path,img)))
        X.append(img_pth_list)

    return X,Y,cls_int_map

def XY_dataset_to_episodes(X: List[List[str]], Y: List[int],\
    N_way: int= 5, K_shot: int= 1,  query_samples_per_class: int= 2) -> \
         List[Tuple[Tuple[List[str],List[int]],Tuple[List[str],List[int]]]]:
    """Converts X and Y into list of episodes for few shot learning paradigm

    Parameters
    ----------
        X: List[List[str]]
            Contains list of image file paths per class
        Y: List[int]
            Contains list of classes
        N_way: int, optional (default = 5)
            No of classes per episode
        K_shot: int, optional (default = 1)
            No of samples per class for support set
        query_samples_per_class: int, optional (default = 2)
            No of query samples per class

    Returns
    -------
    List[Tuple[Tuple[List[str],List[int]],Tuple[List[str],List[int]]]]
        list of episodes, where each episode consists of tuple of support set 
        and query set. Each set consists of list of image paths and their labels    
    """
    X = copy.deepcopy(X); Y = copy.deepcopy(Y)
    episodes = []
    
    #classes which have samples >= (K_shot + query_samples_per_class) .It updates during dequeue operations.
    active_classes = set() 

    for i , img_paths_per_class in enumerate(X):
        if len(img_paths_per_class)>= (K_shot + query_samples_per_class):
            active_classes.add(i)

    #in each iteration sample N_way classes from active classes and pick K_shot 
    #samples per class for support set and query_samples_per_class for query set    
    while(len(active_classes)>=N_way):
        episode_cls_idxs = list(np.random.choice(list(active_classes),size=N_way,replace=False))
        support_set_img_paths = [] ; support_set_labels = []
        query_set_img_paths = [] ; query_set_labels = []

        for idx in episode_cls_idxs:
            support_set_img_paths.append(X[idx][:K_shot])
            support_set_labels.append(Y[idx])
            query_set_img_paths.append(X[idx][K_shot:K_shot+query_samples_per_class])
            query_set_labels.append(Y[idx])
            X[idx] = X[idx][K_shot+query_samples_per_class:]
        
            #check which classes still have >=K_shot elements
            if len(X[idx])<K_shot + query_samples_per_class:
                active_classes.remove(idx)
        

        support_set_img_paths = np.array(support_set_img_paths).flatten().tolist()
        query_set_img_paths = np.array(query_set_img_paths).flatten().tolist()
        support_set_labels = np.repeat(support_set_labels,K_shot).tolist()
        query_set_labels = np.repeat(query_set_labels,query_samples_per_class).tolist()

        
        episodes.append(((support_set_img_paths,support_set_labels),(query_set_img_paths,query_set_labels)))
    
    return episodes

            





    
    




