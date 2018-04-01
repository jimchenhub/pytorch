import math
import torch
from .optimizer import Optimizer


class AdamDelta(Optimizer):
    """Implements AdamDelta algorithm.

    Add RMS part to momentum 1 unit.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)


    """

    def __init__(self, params, lr=1.0, betas=(0.9, 0.9, 0.99), eps=1e-8, 
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamDelta, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamDelta, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamDelta does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # delta of all
                    state['acc_delta'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    

                exp_avg, acc_delta, exp_avg_sq = state['exp_avg'], state['acc_delta'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2, beta3 = group['betas']
                eps = group['eps']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta3).addcmul_(1 - beta3, grad, grad)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    std = max_exp_avg_sq.add(eps).sqrt_()
                else:
                    std = exp_avg_sq.add(eps).sqrt_()
                
                delta = acc_delta.add(eps).sqrt_().div_(std).mul_(exp_avg)


                # bias_correction2 = 1 - beta3 ** state['step']
                # if amsgrad:
                #     step_size = group['lr'] * math.sqrt(bias_correction2)
                # else:
                #     bias_correction1 = 1 - beta1 ** state['step']
                #     step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                step_size = group['lr']
                p.data.add_(-step_size, delta)

                acc_delta.mul_(beta2).addcmul_(1 - beta2, delta, delta)

        return loss
