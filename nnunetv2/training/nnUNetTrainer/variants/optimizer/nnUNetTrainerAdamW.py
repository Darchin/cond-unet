import math

import torch

from nnunetv2.training.nnUNetTrainer.variants.benchmarking.nnUNetTrainer_Computation_Metrics import nnUNetTrainerComputationMetrics


class LinearWarmupCosineAnnealingLR:
    def __init__(self, optimizer, initial_lr: float, warmup_epochs: int, num_epochs: int, min_lr: float):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.num_epochs = num_epochs
        self.min_lr = min_lr
        self.ctr = 0
        self._last_lr = [self._compute_lr(0) for _ in optimizer.param_groups]

        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr

    def _compute_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            progress = epoch / (self.warmup_epochs - 1)
            return self.initial_lr * (0.1 + 0.9 * progress)

        progress = (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
        progress = min(max(progress, 0), 1)
        return self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self._compute_lr(current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr


class nnUNetTrainerAdamW(nnUNetTrainerComputationMetrics):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        self.initial_lr = 3e-4
        self.weight_decay = 1e-3
        self.num_epochs = 250
        self.warmup_epochs = 5
        self.min_lr = 1e-6
        self.enable_deep_supervision = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(),
                                      lr=self.initial_lr,
                                      weight_decay=self.weight_decay)
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, self.initial_lr, self.warmup_epochs, self.num_epochs, self.min_lr
        )
        return optimizer, lr_scheduler
