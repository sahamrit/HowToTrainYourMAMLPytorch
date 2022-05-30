"""TODO
"""

from tokenize import String
import torch 
import torch.nn as nn
import numpy as np
import os 
import copy

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchvision.models.resnet import ResNet
from typing import *
from tqdm import tqdm
from traitlets import Bool

#TODO
def maml_inner_loop_train(loss_fn: nn.modules.loss._Loss, optimizer: torch.optim.Optimizer,\
    model: ResNet, xs: torch.Tensor, ys: torch.Tensor, N_way: int, inner_loop_steps: int):
    """TODO
    """

    model.train()

    for i in range(inner_loop_steps):
        
        optimizer.zero_grad()
        y_pred = model(xs)
        loss = loss_fn(y_pred, torch.eye(N_way)[ys].to(ys.device))
        grads = torch.autograd.grad(loss,model.parameters(),create_graph=True)
        named_grads = {}
        for (name , param) , grad in zip(model.named_parameters(),grads):
            named_grads[name] = grad
            setattr(param,"param_name",name)
        state_dict = model.state_dict()
        state_dict.update(optimizer.step(named_grads))
        model.load_state_dict(state_dict)

def run_episodes(is_train:Bool ,loss_fn: nn.modules.loss._Loss, optimizer: torch.optim.Optimizer,\
    optimizer_inner_loop: Optimizer,model: ResNet, xs: torch.Tensor, xq: torch.Tensor,\
        ys: torch.Tensor, yq: torch.Tensor, N_way: int, inner_loop_steps: int): 
    """TODO
    """

    query_set_loss = 0.; correct_preds = 0; total_preds = 0

    for tasks in range(xs.shape[0]):
        support_set_images = xs[tasks]
        _support_set_labels = ys[tasks]
        
        #make the labels from 0 to no_of_unique_labels
        unique_labels, support_set_labels = torch.unique(\
            _support_set_labels, sorted= True, return_inverse=True)
                    
        #have a model copy for each task
        new_model = copy.deepcopy(model)
        new_optimizer = optimizer_inner_loop(new_model.parameters(), lr=optimizer.defaults['lr']) # TODO fix LR

        maml_inner_loop_train(loss_fn, new_optimizer, new_model, support_set_images,\
            support_set_labels,N_way, inner_loop_steps)
        
        with torch.set_grad_enabled(is_train):

            query_set_preds = new_model(xq[tasks])
            _query_set_labels = yq[tasks]

            #make the labels from 0 to no_of_unique_labels
            _, query_set_labels = torch.unique(\
                _query_set_labels, sorted= True, return_inverse=True)

            query_set_loss += loss_fn(query_set_preds,\
                torch.eye(N_way)[query_set_labels].to(query_set_labels.device))

            with torch.no_grad():
                _ , preds = torch.max(query_set_preds.data,1)
                correct_preds += (preds == query_set_labels).sum().item()
                total_preds += preds.shape[0]

    return query_set_loss, correct_preds, total_preds
    

#TODO add type hinting here.
def do_train(iter:int , epoch:int , device: String, loss_fn: nn.modules.loss._Loss, optimizer: Optimizer,optimizer_inner_loop: Optimizer,\
    model: ResNet, train_dl: DataLoader, val_dl: DataLoader , inner_loop_steps: int, train_verbosity: int, \
        val_freq: int)-> torch.float64:
    """Trains MAML Model for one epoch. It also has validations at interval
    of steps

    Parameters
    ----------
    iter,epoch,inner_loop_steps,train_verbosity,val_freq: TODO
    device: TODO

    loss_fn: TODO

    optimizer: torch.optim.Optimizer
        One of many torch optimizers like SGD, Adam etc.

    optimizer_inner_loop: TODO

    train_dl: torch.utils.data.DataLoader
        Dataloader for training set. Consists of (xs,ys,xq,yq) 
        where s denotes support set and q denotes query set.
        xs&xq are of shape (batch_sz,support_sz,C,H,W)
    val_dl: torch.utils.data.DataLoader
        Same as train_dl withonly difference of data coming from
        validation set. This is required when one epoch is too huge 
        and you want to validate intermediate steps
    model: torchvision.models.resnet.ResNet
        Model of ResNet family with the fc layer changed to having last
        dimension = N_way

    Returns
    -------

    torch.float64
        Aggregated Validation Loss
        TODO: what aggregation ?
    """

    model.train()
    optimizer.zero_grad()

    #TODO can we make sure if i stop a model inbetween i start from the same
    #iteration


    for (xs,ys), (xq,yq) in tqdm(train_dl):
        #TODO: spawn multiple processes here

        xs = xs.to(device); xq = xq.to(device); ys = ys.to(device); yq = yq.to(device)
        iter += 1

        query_set_loss, correct_preds, total_preds = run_episodes(True, loss_fn\
            ,optimizer, optimizer_inner_loop, model, xs, xq\
                ,ys ,yq, train_dl.dataset.N, inner_loop_steps)

        query_set_loss/=xs.shape[0]

        if iter% train_verbosity == 0:
            print(f"Training: Epoch: {epoch} Iteration: {iter} Loss => {query_set_loss:.4f} with batch size => {xs.shape[0]}")
            print(f"Training: Epoch: {epoch} Iteration: {iter} Accuracy => {correct_preds*100/total_preds:.2f}% with total_preds => {total_preds} ")

        query_set_loss.backward()
        optimizer.step()

        if iter% val_freq == 0:
            model.eval()
            optimizer.zero_grad()

            val_loss = []; correct_preds= 0; total_preds = 0

            for (xs,ys), (xq,yq) in tqdm(val_dl):
                #TODO: spawn multiple processes here

                xs = xs.to(device); xq = xq.to(device); ys = ys.to(device); yq = yq.to(device)

                query_set_loss, _correct_preds, _total_preds = run_episodes(False, loss_fn\
                    ,optimizer, optimizer_inner_loop, model, xs, xq\
                        ,ys ,yq, val_dl.dataset.N, inner_loop_steps)

                query_set_loss/=xs.shape[0]

                val_loss.append(query_set_loss.cpu()); correct_preds += _correct_preds
                total_preds += _total_preds

            print(f"Validation: Epoch: {epoch} Iteration: {iter} Loss => {np.mean(val_loss):.4f} with batch size => {xs.shape[0]}")
            print(f"Validation: Epoch: {epoch} Iteration: {iter} Accuracy => {correct_preds*100/total_preds:.2f}% with total_preds => {total_preds} ")

            model.train()
            optimizer.zero_grad()

        #TODO Can we keep a loss global list to update the losses regularly 
        #TODO Use Logger
        #TODO Implement ckpting
        #TODO Implement validation Loop
        #TODO where to put model.eval() ??


