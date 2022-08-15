# Open source specific imports
from typing import *
import torch
import torch.nn as nn

import copy

# Local code specific imports


def run_episodes(
    is_train: bool,
    loss_fn: nn.modules.loss._Loss,
    inner_loop_lr: float,
    optimizer_inner_loop: torch.optim.Optimizer,
    model: nn.Module,
    xs: torch.Tensor,
    xq: torch.Tensor,
    ys: torch.Tensor,
    yq: torch.Tensor,
    N_way: int,
    inner_loop_steps: int,
) -> Tuple[torch.Tensor, int, int]:
    """Run episodes for a batch of tasks which includes inner and outer loop training

     Parameters
     ----------
     is_train:bool
         Decides if meta model is updated or not.
     loss_fn: nn.modules.loss._Loss
         torch.nn.BCEWithLogitsLoss in specific
    inner_loop_lr: float
         LR of inner loop optimizer
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

    total_query_set_loss = 0.0
    total_correct_preds = 0
    total_preds = 0

    for tasks in range(xs.shape[0]):

        support_set_images = xs[tasks]
        _support_set_labels = ys[tasks]

        # make the labels from 0 to no_of_unique_labels
        _, support_set_labels = torch.unique(
            _support_set_labels, sorted=True, return_inverse=True
        )

        # have a model copy for each task
        new_model = clone_model(model)
        new_optimizer = optimizer_inner_loop(
            new_model.parameters(), lr=inner_loop_lr
        )  # TODO fix LR

        maml_inner_loop_train(
            loss_fn,
            new_optimizer,
            new_model,
            support_set_images,
            support_set_labels,
            N_way,
            inner_loop_steps,
        )

        query_set_loss, correct_preds, no_preds = evaluate_query_set(
            new_model, xq[tasks], yq[tasks], is_train, loss_fn
        )
        total_query_set_loss += query_set_loss
        total_correct_preds += correct_preds
        total_preds += no_preds

    return total_query_set_loss, total_correct_preds, total_preds


def maml_inner_loop_train(
    loss_fn: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    N_way: int,
    inner_loop_steps: int,
) -> None:
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
        grads = torch.autograd.grad(
            loss, yeild_params(old_param_dict), create_graph=True
        )
        model, new_param_dict = update_model(model, grads, old_param_dict, optimizer)
        old_param_dict = new_param_dict


def yeild_params(named_params: dict) -> List:
    params = []
    for name, p in named_params.items():
        params.append(p)
    return params


def evaluate_query_set(
    model: nn.Module,
    xq_per_task: torch.Tensor,
    yq_per_task: torch.Tensor,
    is_train: bool,
    loss_fn: nn.modules.loss._Loss,
) -> Tuple[torch.Tensor, int, int]:
    """Evaluates Query Set Loss, Correct Predictions and Total Predictions.
    When is_train is enabled the process is tracked for backprop

    Parameters
    ----------

    model: nn.Module
        Model of ResNet family with the fc layer changed to having last dimension = N_way
    xq_per_task: torch.Tensor
        Query set images for the task
        shape -> [N_way*query_samples_per_class,C,H,W]
    yq_per_task: torch.Tensor
        Query set labels for the task. Note that the actual labels are mapped from 0 to
        N_way consistent across support and query set
        shape -> [N_way*query_samples_per_class]
    is_train:Bool
        Decides if meta model is updated or not.
    loss_fn: nn.modules.loss._Loss
        torch.nn.BCEWithLogitsLoss in specific

    Returns
    -------
    Tuple[torch.Tensor, int, int]
        (Query Set Loss, Correct Predictions in Query Set, Total Predicitions in Query Set)
        Here Query Set is per Task

    """

    with torch.set_grad_enabled(is_train):

        query_set_preds = model(xq_per_task)
        _query_set_labels = yq_per_task

        # make the labels from 0 to no_of_unique_labels
        _, query_set_labels = torch.unique(
            _query_set_labels, sorted=True, return_inverse=True
        )

        query_set_loss = loss_fn(
            query_set_preds,
            torch.eye(query_set_preds.shape[-1])[query_set_labels].to(
                query_set_labels.device
            ),
        )

        with torch.no_grad():
            _, preds = torch.max(query_set_preds.data, 1)
            correct_preds = (preds == query_set_labels).sum().item()
            total_preds = preds.shape[0]

    return query_set_loss, correct_preds, total_preds


def clone_model(model: nn.Module) -> nn.Module:
    """Clones the Model and Sets the name of params in attribute 'param_name'

    Parameters
    ----------
    model: nn.Module
        Model of ResNet family with the fc layer changed to having last dimension = N_way

    Returns
    -------
    nn.Module
        Cloned Model
    """

    new_model = copy.deepcopy(model)
    load_param_dict(new_model, OrderedDict(list(model.named_parameters())))

    for name, param in new_model.named_parameters():
        param.param_name = name

    return new_model


def update_model(model, grads, param_dict, optimizer):
    """Update the model parameters by gradient descent using the grads

    Parameters
    ----------
    model: nn.Module
        Model of ResNet family with the fc layer changed to having last dimension = N_way

    grads: Tuple[torch.Tensor, ...]
        Gradients of all the model parameters

    param_dict: dict[str,torch.Tensor]
        Dictionary of all model parameters with key as param name and value as param itself

    optimizer: torch.optim.Optimizer
        engine.optimizers.AdamExplicitGrad which takes specific grad parameter to avoid
        interaction with p.grad. Needed for higher order derivates

    Returns
    -------
    Tuple[nn.Module, dict[str,torch.Tensor]]
        The new model and updated param dictionary
    """

    named_grads = {}

    for (name, _), grad in zip(param_dict.items(), grads):
        named_grads[name] = grad

    new_param_dict = param_dict
    updated_params = optimizer.step(named_grads)

    new_param_dict.update(updated_params)
    load_param_dict(model, new_param_dict)

    return model, new_param_dict


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
        del_attr(model, name.split("."))
        set_attr(model, name.split("."), param)


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
