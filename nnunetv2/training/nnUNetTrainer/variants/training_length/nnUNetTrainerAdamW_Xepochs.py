import torch

from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerAdamW import nnUNetTrainerAdamW


class nnUNetTrainerAdamW_250epochs(nnUNetTrainerAdamW):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        
class nnUNetTrainerAdamW_1000epochs(nnUNetTrainerAdamW):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000