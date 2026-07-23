import math

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.network_architecture.cond_unet import Router


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


class nnUNetTrainerAdamW(nnUNetTrainer):
    configurable_trainer_keys = {
        'initial_lr',
        'weight_decay',
        'num_epochs',
        'warmup_epochs',
        'min_lr',
        'enable_deep_supervision',
        'expert_dropout_anneal_epochs',
    }

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 3e-4
        self.weight_decay = 1e-3
        self.num_epochs = 250
        self.warmup_epochs = 5
        self.min_lr = 1e-6
        self.enable_deep_supervision = False
        self.expert_dropout_anneal_epochs = None
        self._apply_trainer_configuration()

    @staticmethod
    def _require_real(value, name: str, min_value: float = None, allow_zero: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"trainer.{name} must be a number, got {type(value).__name__}")
        value = float(value)
        if min_value is not None:
            if allow_zero:
                valid = value >= min_value
            else:
                valid = value > min_value
            if not valid:
                cmp = ">=" if allow_zero else ">"
                raise ValueError(f"trainer.{name} must be {cmp} {min_value}, got {value}")
        return value

    @staticmethod
    def _require_int(value, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"trainer.{name} must be an integer, got {type(value).__name__}")
        return value

    def _apply_trainer_configuration(self):
        trainer_config = self.configuration_manager.trainer
        if not trainer_config:
            return
        if not isinstance(trainer_config, dict):
            raise TypeError(f"trainer must be a dict, got {type(trainer_config).__name__}")
        unknown_keys = set(trainer_config) - self.configurable_trainer_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown trainer configuration keys for {self.__class__.__name__}: {sorted(unknown_keys)}"
            )

        if 'initial_lr' in trainer_config:
            self.initial_lr = self._require_real(trainer_config['initial_lr'], 'initial_lr', 0)
        if 'weight_decay' in trainer_config:
            self.weight_decay = self._require_real(trainer_config['weight_decay'], 'weight_decay', 0, allow_zero=True)
        if 'num_epochs' in trainer_config:
            self.num_epochs = self._require_int(trainer_config['num_epochs'], 'num_epochs')
        if 'warmup_epochs' in trainer_config:
            self.warmup_epochs = self._require_int(trainer_config['warmup_epochs'], 'warmup_epochs')
        if 'min_lr' in trainer_config:
            self.min_lr = self._require_real(trainer_config['min_lr'], 'min_lr', 0, allow_zero=True)
        if 'enable_deep_supervision' in trainer_config:
            if not isinstance(trainer_config['enable_deep_supervision'], bool):
                raise TypeError(
                    f"trainer.enable_deep_supervision must be a bool, got "
                    f"{type(trainer_config['enable_deep_supervision']).__name__}"
                )
            self.enable_deep_supervision = trainer_config['enable_deep_supervision']
        if 'expert_dropout_anneal_epochs' in trainer_config:
            value = trainer_config['expert_dropout_anneal_epochs']
            if value is not None:
                value = self._require_int(value, 'expert_dropout_anneal_epochs')
                if value <= 0:
                    raise ValueError(
                        "trainer.expert_dropout_anneal_epochs must be a positive "
                        f"integer or None, got {value}"
                    )
            self.expert_dropout_anneal_epochs = value

        if self.warmup_epochs < 2:
            raise ValueError(
                f"trainer.warmup_epochs must be >= 2 for the AdamW warmup schedule, got {self.warmup_epochs}"
            )
        if self.num_epochs <= self.warmup_epochs:
            raise ValueError(
                f"trainer.num_epochs must be greater than trainer.warmup_epochs, got "
                f"{self.num_epochs} and {self.warmup_epochs}"
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(),
                                      lr=self.initial_lr,
                                      weight_decay=self.weight_decay)
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, self.initial_lr, self.warmup_epochs, self.num_epochs, self.min_lr
        )
        return optimizer, lr_scheduler

    def _update_expert_dropout(self) -> None:
        if self.expert_dropout_anneal_epochs is None:
            return
        factor = max(
            0.0,
            1.0 - self.current_epoch / self.expert_dropout_anneal_epochs,
        )
        for module in self.network.modules():
            if isinstance(module, Router):
                module.set_expert_dropout(module.expert_dropout * factor)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._update_expert_dropout()
