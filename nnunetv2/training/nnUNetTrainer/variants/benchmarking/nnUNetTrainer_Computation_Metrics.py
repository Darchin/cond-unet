import time
import torch
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerComputationMetrics(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Raw metrics history across all epochs
        self.compute_history = {
            "epoch_times": [],
            "train_step_times": [],
            "train_memory_gb": [],
            "val_time_per_sample": []
        }

    def run_training(self):
        self.on_train_start()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            
            # --- 1. Average Time Per Epoch (Start Track) ---
            epoch_start_time = time.time()
            
            self.on_train_epoch_start()
            
            # Reset peak memory stats at the start of the epoch to capture this epoch's max
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            
            # --- 2. Average Time Per Step (Training) ---
            train_outputs = []
            train_start_time = time.time()
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            train_duration = time.time() - train_start_time
            
            avg_step_time = train_duration / self.num_iterations_per_epoch
            self.compute_history["train_step_times"].append(avg_step_time)
            
            # --- 3. Memory Usage Per Step (GB) ---
            if self.device.type == 'cuda':
                peak_mem_bytes = torch.cuda.max_memory_allocated(self.device)
                peak_mem_gb = peak_mem_bytes / (1024 ** 3)
            else:
                peak_mem_gb = 0.0
            self.compute_history["train_memory_gb"].append(peak_mem_gb)
            
            self.on_train_epoch_end(train_outputs)
            
            # --- 4. Average Time Per Sample (Validation) ---
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                val_start_time = time.time()
                total_val_samples = 0
                
                for batch_id in range(self.num_val_iterations_per_epoch):
                    batch = next(self.dataloader_val)
                    # Dynamically capture batch size to handle uneven split strategies
                    total_val_samples += batch['data'].shape[0] 
                    val_outputs.append(self.validation_step(batch))
                    
                val_duration = time.time() - val_start_time
                avg_val_time_per_sample = val_duration / total_val_samples if total_val_samples > 0 else 0
                self.compute_history["val_time_per_sample"].append(avg_val_time_per_sample)
                
                self.on_validation_epoch_end(val_outputs)
            
            self.on_epoch_end()
            
            # --- 1. Average Time Per Epoch (End Track) ---
            epoch_duration = time.time() - epoch_start_time
            self.compute_history["epoch_times"].append(epoch_duration)
            
            # --- 5. Save to compute.json ---
            if self.local_rank == 0:
                summary = {
                    "running_averages": {
                        "avg_time_per_epoch_seconds": float(np.mean(self.compute_history["epoch_times"])),
                        "avg_time_per_step_seconds": float(np.mean(self.compute_history["train_step_times"])),
                        "avg_memory_allocated_gb": float(np.mean(self.compute_history["train_memory_gb"])),
                        "avg_time_per_val_sample_seconds": float(np.mean(self.compute_history["val_time_per_sample"]))
                    },
                    "epoch_by_epoch_history": self.compute_history
                }
                save_json(summary, join(self.output_folder, "compute.json"), sort_keys=False)
                
        self.on_train_end()