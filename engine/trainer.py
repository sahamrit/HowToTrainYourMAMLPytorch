"""This module contains training engine for Model Agnostic Meta Learning paper implementation

This file contains the following functions:
    *maml_inner_loop_train : Responsible for training of each task called fast learning 
        (trying to acquire task specific knowledge)
    *run_episodes: Calls maml_inner_loop_train for a batch of task learning and then 
        updates the meta model (acquires task agnostic knowledge)
    *do_train: Conducts training and validation loop for the entire dataset 
"""
import torchviz
import torch 
import torch.nn as nn
import numpy as np
import os 
import copy
import wandb

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchvision.models.resnet import ResNet
from typing import *
from tqdm import tqdm
from traitlets import Bool
from torchviz import make_dot

from configs import *
from utils import set_attr, del_attr, load_param_dict, yeild_params
from utils.training_utils import clone_model, evaluate_query_set, update_model



def maml_inner_loop_train(loss_fn: nn.modules.loss._Loss, optimizer: torch.optim.Optimizer,\
    model: nn.Module, xs: torch.Tensor, ys: torch.Tensor, N_way: int, inner_loop_steps: int) -> None:
    """Implementation of inner training loop of MAML. Responsible for task specific knowledge

    Parameters
    ----------
    loss_fn: nn.modules.loss._Loss
        torch.nn.BCEWithLogitsLoss in specific
    optimizer: torch.optim.Optimizer
        engine.optimizers.AdamExplicitGrad which takes specific grad parameter to avoid
        interaction with p.grad. Needed for higher order derivates
    model: nn.Module
        Model of ResNet family with the fc layer changed to having last dimension = N_way
    xs: torch.Tensor
        Images from the support set of a particular task
        xs.shape -> [N_way*K_shot,C,H,W]
    ys: torch.Tensor
        Labels from the support set of a particular task. Note that the actual labels are mapped from 0 to 
        N_way consistent across support and query set
        ys.shape -> [N_way*K_shot]
    N_way: int
        N in N-way-K-shot learning
    inner_loop_steps: int
        No of training steps for inner loop so that the model adapts sufficient to the
        new task  

    Returns
    -------
    None 
    """

    model.train()
    old_param_dict = OrderedDict(list(model.named_parameters()))
    for i in range(inner_loop_steps):
        
        optimizer.zero_grad()
        y_pred = model(xs)
        loss = loss_fn(y_pred, torch.eye(N_way)[ys].to(ys.device))
        grads = torch.autograd.grad(loss,yeild_params(old_param_dict),create_graph=True)
        model, new_param_dict = update_model(model, grads, old_param_dict, optimizer)
        old_param_dict = new_param_dict

def run_episodes(is_train:Bool ,loss_fn: nn.modules.loss._Loss, optimizer: torch.optim.Optimizer,\
    optimizer_inner_loop: Optimizer,model: nn.Module, xs: torch.Tensor, xq: torch.Tensor,\
        ys: torch.Tensor, yq: torch.Tensor, N_way: int, inner_loop_steps: int) -> Tuple[torch.Tensor, int, int]: 
    """Run episodes for a batch of tasks which includes inner and outer loop training

    Parameters
    ----------
    is_train:Bool
        Decides if meta model is updated or not.
    loss_fn: nn.modules.loss._Loss
        torch.nn.BCEWithLogitsLoss in specific
    optimizer: torch.optim.Optimizer
        Any of torch optimizers for the meta model
    optimizer_inner_loop: torch.optim.Optimizer
        Optimizer for the inner loop
        engine.optimizers.AdamExplicitGrad which takes specific grad parameter to avoid
        interaction with p.grad. Needed for higher order derivates
    model: nn.Module
        Model of ResNet family with the fc layer changed to having last dimension = N_way
    xs: torch.Tensor
        Batch of support set images
        xs.shape -> [batch_size,N_way*K_shot,C,H,W]
    ys: torch.Tensor
        Batch of support set labels. Note that the actual labels are mapped from 0 to 
        N_way consistent across support and query set  
        ys.shape -> [batch_size,N_way*K_shot]
    xq: torch.Tensor
        Batch of query set images
        xs.shape -> [batch_size,N_way*query_samples_per_class,C,H,W]
    yq: torch.Tensor
        Batch of query set labels. Note that the actual labels are mapped from 0 to 
        N_way consistent across support and query set
        ys.shape -> [batch_size,N_way*query_samples_per_class]
    N_way: int
        N in N-way-K-shot learning
    inner_loop_steps: int
        No of training steps for inner loop so that the model adapts sufficient to the
        new task

    Returns
    -------
    Tuple[torch.Tensor, int, int]
        (Total Query Set Loss, Correct Predictions in Query Set, Total Predicitions in Query Set)  
        Here Query Set includes all the tasks in the Batch  
    """

    total_query_set_loss = 0.; total_correct_preds = 0; total_preds = 0

    for tasks in range(xs.shape[0]):
        
        support_set_images = xs[tasks];  _support_set_labels = ys[tasks]
        
        #make the labels from 0 to no_of_unique_labels
        _ , support_set_labels = torch.unique(\
            _support_set_labels, sorted= True, return_inverse=True)
       
        #have a model copy for each task
        new_model = clone_model(model)
        new_optimizer = optimizer_inner_loop(new_model.parameters(), lr=optimizer.defaults['inner_lr']) # TODO fix LR


        maml_inner_loop_train(loss_fn, new_optimizer, new_model, support_set_images,\
            support_set_labels,N_way, inner_loop_steps)

        query_set_loss, correct_preds, no_preds =  evaluate_query_set(new_model, xq[tasks], yq[tasks], is_train, loss_fn )        
        total_query_set_loss += query_set_loss; total_correct_preds += correct_preds; total_preds += no_preds

    return total_query_set_loss, total_correct_preds, total_preds
    
def do_train(iter:int , epoch:int , device: str, loss_fn: nn.modules.loss._Loss, optimizer: Optimizer,optimizer_inner_loop: Optimizer,\
    model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, conf: ExperimentConfig )-> None:
    """Trains MAML Model for one epoch. It also has validates the model at regular intervals

    Parameters
    ----------
    iter:int
        global parameter to monitor the total number of iterations
    epoch:int
        global parameter to monitor current epoch
    device: str
        cpu or cuda:0
    loss_fn: nn.modules.loss._Loss
        torch.nn.BCEWithLogitsLoss in specific
    optimizer: torch.optim.Optimizer
        Any of torch optimizers for the meta model
    optimizer_inner_loop: torch.optim.Optimizer
        Optimizer for the inner loop
        engine.optimizers.AdamExplicitGrad which takes specific grad parameter to avoid
        interaction with p.grad. Needed for higher order derivates
    model: nn.Module
        Model of ResNet family with the fc layer changed to having last dimension = N_way
    train_dl: torch.utils.data.DataLoader
        Dataloader for training set. Consists of (xs,ys,xq,yq) 
        where s denotes support set and q denotes query set.
        xs&xq are of shape (batch_sz,support_sz,C,H,W)
    val_dl: torch.utils.data.DataLoader
        Same as train_dl with only difference of data coming from
        validation set. This is required when one epoch is too huge 
        and you want to validate intermediate steps
    conf: ExperimentConfig
        centralised config for the experiment

    Returns
    -------
    None
    """

    model.train()
    

    #TODO can we make sure if i stop a model inbetween i start from the same
    #iteration


    for (xs,ys), (xq,yq) in tqdm(train_dl,disable=True):
        #TODO: spawn multiple processes here

        optimizer.zero_grad()

        xs = xs.to(device); xq = xq.to(device); ys = ys.to(device); yq = yq.to(device)
        iter += 1

        query_set_loss, correct_preds, total_preds = run_episodes(True, loss_fn\
            ,optimizer, optimizer_inner_loop, model, xs, xq\
                ,ys ,yq, train_dl.dataset.N, conf.training.inner_loop_steps)

        # if iter == 1:
        #     make_dot(query_set_loss, params=dict(list(model.named_parameters())) )\
        #         .render(filename = os.path.join(conf.log_dir,conf.curr_run), format = 'png')


        query_set_loss/=xs.shape[0]

        if conf.training.wandb_logging:
            wandb.log({"trainingQuerySetLossPerMinibatch":query_set_loss, "trainingQuerySetAccuracyPerMinibatch":correct_preds*100/total_preds})

        if iter% conf.training.train_verbosity == 0:
            print(f"Training: Epoch: {epoch} Iteration: {iter} Loss => {query_set_loss:.4f} with batch size => {xs.shape[0]}")
            print(f"Training: Epoch: {epoch} Iteration: {iter} Accuracy => {correct_preds*100/total_preds:.2f}% with total_preds => {total_preds} ")

        query_set_loss.backward()
        optimizer.step()

        if iter% conf.training.val_freq == 0:
            model.eval()

            val_loss = []; correct_preds= 0; total_preds = 0

            for (xs,ys), (xq,yq) in tqdm(val_dl,disable=True):
                #TODO: spawn multiple processes here

                xs = xs.to(device); xq = xq.to(device); ys = ys.to(device); yq = yq.to(device)

                query_set_loss, _correct_preds, _total_preds = run_episodes(False, loss_fn\
                    ,optimizer, optimizer_inner_loop, model, xs, xq\
                        ,ys ,yq, val_dl.dataset.N, conf.training.inner_loop_steps)

                query_set_loss/=xs.shape[0]

                val_loss.append(query_set_loss.cpu()); correct_preds += _correct_preds
                total_preds += _total_preds

            if conf.training.wandb_logging:
                wandb.log({"valLoss":np.mean(val_loss), "valAccuracy":correct_preds*100/total_preds})

            print(f"Validation: Epoch: {epoch} Iteration: {iter} Loss => {np.mean(val_loss):.4f} with batch size => {xs.shape[0]}")
            print(f"Validation: Epoch: {epoch} Iteration: {iter} Accuracy => {correct_preds*100/total_preds:.2f}% with total_preds => {total_preds} ")


            model.train()

        #TODO Can we keep a loss global list to update the losses regularly 
        #TODO Use Logger
        #TODO Implement ckpting
        #TODO Implement validation Loop
        #TODO where to put model.eval() ??


