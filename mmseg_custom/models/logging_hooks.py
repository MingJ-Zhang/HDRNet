from __future__ import annotations

import torch
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS


@HOOKS.register_module()
class DetailedTrainingLogHook(Hook):
    """Extra training logger for LR, grad norm and CUDA memory.

    This hook is intentionally lightweight and can be enabled together with the
    default MMEngine LoggerHook.
    """

    priority = 'LOW'

    def __init__(self, interval: int = 50, log_grad_norm: bool = True):
        self.interval = interval
        self.log_grad_norm = log_grad_norm

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        cur_iter = runner.iter + 1
        if cur_iter % self.interval != 0:
            return

        lr = None
        if runner.optim_wrapper is not None:
            lr_dict = runner.optim_wrapper.get_lr()
            if isinstance(lr_dict, dict) and lr_dict:
                first_group = next(iter(lr_dict.values()))
                lr = first_group[0] if isinstance(first_group, list) else first_group

        grad_norm = None
        if self.log_grad_norm and hasattr(runner.model, 'parameters'):
            total_sq = 0.0
            for p in runner.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2).item()
                    total_sq += param_norm ** 2
            grad_norm = total_sq ** 0.5

        mem = 0.0
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1024 / 1024

        msg = f'[DetailedLog] iter={cur_iter}'
        if lr is not None:
            msg += f', lr={lr:.6e}'
        if grad_norm is not None:
            msg += f', grad_norm={grad_norm:.4f}'
        msg += f', cuda_max_mem={mem:.1f}MB'
        print_log(msg, logger='current')
