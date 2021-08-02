import torch
import math

###################################################
#     Two equivalent manual adam implementations
###################################################


class AdamOptimizer:
    def __init__(self, x: torch.Tensor):
        self.m = torch.zeros_like(x)
        self.v = torch.zeros_like(x)
        self.t = 0

    def __call__(
        self,
        gradient: torch.Tensor,
        stepsize: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        self.t += 1

        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2

        bias_correction_1 = 1 - beta1 ** self.t
        bias_correction_2 = 1 - beta2 ** self.t

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2

        return -stepsize * m_hat / (torch.sqrt(v_hat) + epsilon)


class Adam_Manual:
    def __init__(self, x: torch.Tensor):
        self.exp_avgs = torch.zeros_like(x)
        self.exp_avg_sqs = torch.zeros_like(x)
        self.steps = 0

    def __call__(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ) -> torch.Tensor:

        self.steps += 1

        exp_avg = self.exp_avgs
        exp_avg_sq = self.exp_avg_sqs
        step = self.steps

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)
