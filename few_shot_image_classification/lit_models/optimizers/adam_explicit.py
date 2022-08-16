"""This script contains custom optimizer implementation which takes in gradients
as explicit paramater and returns model state_dict after optimizer.step()

This file contains the following functions:
    *adam: functional form of adam optimizer modified to return state_dict
    *AdamExplicitGrad: Modified Adam Optimizer class to take gradients explicitly
        and return updated model state_dict.
"""
import torch
import torch.optim as optim
from typing import *


class AdamExplicitGrad(optim.Optimizer):
    """Implements Adam algorithm with named_grads passed to step

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)

    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        *,
        maximize: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )
        super(AdamExplicitGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamExplicitGrad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)

    def step(self, named_grads, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        new_param_dict = OrderedDict()
        # new_state = OrderedDict()
        for group in self.param_groups:

            beta1, beta2 = group["betas"]

            new_param_list = []
            for p in group["params"]:

                if named_grads[p.param_name] is not None:

                    if named_grads[p.param_name].is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
                    grad = named_grads[p.param_name]

                    # state = self.state[p]
                    # # Lazy state initialization
                    # if len(state) == 0:
                    #     state['step'] = 0
                    #     # Exponential moving average of gradient values
                    #     state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    #     # Exponential moving average of squared gradient values
                    #     state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    #     if group['amsgrad']:
                    #         # Maintains max of all exp. moving avg. of sq. grad. values
                    #         state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # exp_avg = state['exp_avg']
                    # exp_avg_sq = state['exp_avg_sq']

                    # if group['amsgrad']:
                    #     max_exp_avg_sqs = state['max_exp_avg_sq']

                    # # update the steps for each param group update
                    # state['step'] += 1
                    # # record the step after step update
                    # step = state['step']

                    # grad = grad if not group['maximize'] else -grad

                    # bias_correction1 = 1 - beta1 ** step
                    # bias_correction2 = 1 - beta2 ** step

                    # if group['weight_decay'] != 0:
                    #     grad = grad.add(p, alpha=group['weight_decay'])

                    # # Decay the first and second moment running average coefficient
                    # exp_avg = torch.add(torch.mul(exp_avg,beta1),grad,alpha=1 - beta1 )
                    # exp_avg_sq = torch.addcmul(torch.mul(exp_avg_sq,beta2),grad, grad.conj(), value=1 - beta2)

                    # if group['amsgrad']:
                    #     # Maintains the maximum of all 2nd moment running avg. till now
                    #     max_exp_avg_sqs = torch.maximum(max_exp_avg_sqs, exp_avg_sq, )
                    #     # Use the max. for normalizing running avg. of gradient
                    #     denom = torch.add((max_exp_avg_sqs.sqrt() / math.sqrt(bias_correction2)),group['eps'])
                    #     state['max_exp_avg_sq'] = max_exp_avg_sqs
                    # else:
                    #     denom = torch.add((exp_avg_sq.sqrt() / math.sqrt(bias_correction2)),group['eps'])

                    # state['exp_avg'] = exp_avg
                    # state['exp_avg_sq'] = exp_avg_sq

                    # step_size = group['lr'] / bias_correction1
                    # param_new = torch.addcdiv(p,exp_avg,denom,value = -step_size)
                    param_new = p - grad * group["lr"]

                    new_param_dict[p.param_name] = param_new
                    param_new.param_name = p.param_name

                    new_param_list.append(param_new)
                    # new_state[param_new] = state

                else:
                    new_param_list.append(p)

            group["params"] = new_param_list

        # self.state = new_state

        return new_param_dict
