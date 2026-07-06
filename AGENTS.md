# Project Overview

- This repository is a research fork of nnU-Net v2 for medical image segmentation experiments.
- The main research direction is architectural experimentation: speed/accuracy ablations, MobileNet-style inverted bottleneck UNet variants, and conditional add-ons such as CondConv-style dense mixture-of-experts blocks.
- Prefer preserving upstream nnU-Net behavior unless a local research feature explicitly changes it.
- The repo uses `uv` for package management and a local `.venv` managed from `.python-version`.
- The target interpreter is the latest Python 3.11 patch release available to `uv`; `pyproject.toml` constrains the project to `>=3.11,<3.12`.
- Install or refresh the environment with `uv sync --extra dev`, then run Python and tests through `uv run ...` or `.venv/bin/python ...` after syncing.

# Working Guidelines

- Read the local implementation before changing behavior. This fork has deliberate changes in planning, plan inheritance, architecture construction, trainer lookup, and validation.
- Keep edits tightly scoped. Avoid broad upstream-style refactors unless they are required for the feature.
- Use focused tests for planner/config/trainer behavior. Existing local coverage lives in `nnunetv2/tests/test_cond_unet_planner_and_config.py` and related tests.
- The worktree may contain user changes. Do not revert unrelated edits.

# Important Local Features

## CondUNet Architecture

- Custom architecture: `nnunetv2/training/network_architecture/cond_unet.py`.
- The architecture uses MobileNet-inspired inverted bottleneck blocks.
- Conditional blocks are modeled as add-ons to the base blocks so the conditional model can act as a superset of the non-conditional architecture for ablation studies.
- Generated CondUNet planner configurations target this class through the `architecture.network_class_name` field.

## CondUNet Planner

- Planner: `nnunetv2/experiment_planning/experiment_planners/cond_unet_planner.py`.
- Default plans identifier: `nnUNetCondUNetPlans`.
- Custom fingerprint extractor: `nnunetv2/experiment_planning/dataset_fingerprint/cond_unet_fingerprint_extractor.py`.
- The fingerprint extractor adds `spacing_percentiles` to `dataset_fingerprint.json`; the planner requires this field and raises a helpful error if the standard extractor was used.
- The planner is intentionally not fully automatic. It emits incomplete base presets that are suitable for preprocessing but still require experiment-specific scale choices before training.

Generated configurations:

- `base`: incomplete shared preprocessing/resampling configuration.
- `2x`, `3x`, and `4x`: inherit from `base` and represent the main downsampling-factor base presets.

Preset behavior:

- `batch_dice` defaults to `True`.
- Generated architecture nonlinearity is `torch.nn.ReLU` with `{"inplace": True}`.
- Base presets include default CondUNet block-depth and expansion-ratio choices; model width remains experiment-specific.
- Stem kernel size is derived from stem stride with `kernel_size = 2 * stride - 1` per axis.
- `patch_size_unit`, `patch_size_unit_mm`, `patch_size_aspect_ratio`, and `median_image_size_in_voxels` are emitted to help users choose manual patch sizes.
- `patch_size_multiplier` is emitted as `null` intentionally. Users must create child configurations that set it before training.
- Presets include `required_for_training` entries for the manual fields that must be completed.

Expected CondUNet planning workflow:

1. Extract the fingerprint with `CondUNetFingerprintExtractor`.
2. Run experiment planning with `CondUNetPlanner`.
3. Run preprocessing on the incomplete generated presets (`2x`, `3x`, `4x`).
4. Add user configurations that inherit from those presets and fill in manual training details.
5. Train using the completed child configurations.

## Phase One Planner

- Additional planner: `PhaseOnePlanner` in `nnunetv2/experiment_planning/experiment_planners/cond_unet_planner.py`.
- Default plans identifier: `nnUNetCondUNetPhaseOnePlans`.
- It builds on `CondUNetPlanner` and adds a small grid of trainable child configurations for the first ablation phase.
- These child configurations inherit the preprocessing, geometry, architecture defaults, and trainer defaults from the base presets, then fill in patch-size multiplier and model-width choices.

## Plan Inheritance And Required Fields

- Plan handling lives in `nnunetv2/utilities/plans_handling/plans_handler.py`.
- Configuration inheritance supports `inherits_from`.
- `architecture.merge_arch_kwargs: true` causes a child architecture's `arch_kwargs` to merge with the inherited `arch_kwargs` instead of replacing them wholesale. Child values take precedence.
- `patch_size` can be derived from `patch_size_unit * patch_size_multiplier`.
- Training validates `required_for_training` fields and raises actionable errors when manual values such as `patch_size_multiplier` or architecture scale fields are missing.

## AdamW Trainer

- Custom trainer: `nnunetv2/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdamW.py`.
- It uses AdamW with linear warm-up followed by cosine annealing.
- The following parameters can be overridden under the configuration's `trainer` key:
  - `initial_lr`
  - `weight_decay`
  - `num_epochs`
  - `warmup_epochs`
  - `min_lr`
  - `enable_deep_supervision`
- Defaults are defined in both the trainer and the CondUNet planner. Keep them in sync if changing default training behavior.

## Training Validation TTA

- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py` and `nnunetv2/run/run_training.py` include local support for disabling test-time augmentation during post-training validation.
- Preserve the training-script `--disable-tta` behavior when touching validation or training entrypoints.

## Benchmark Utility

- CUDA benchmark script: `nnunetv2/run/benchmark.py`.
- Console entrypoint: `nnUNetv2_benchmark`.
- The utility benchmarks model/data configurations with dummy tensors derived from configuration batch size, patch size, and input channels.
- It runs FP16 mixed-precision forward/backward passes with DiceCE loss and reports mean +/- std dev for peak step memory, forward time, backward time, and total time.
- The benchmark requires CUDA and handles CUDA OOM by clearing CUDA memory and exiting with a concise error.

## Batch Training Utility

- Batch training scheduler: `nnunetv2/run/batch_train.py`.
- Console entrypoint: `nnUNetv2_train_batch`.
- It schedules one `nnUNetv2_train`-equivalent worker per visible GPU and starts queued jobs as GPUs become free.
- Job order is configuration-major, then fold-minor: all requested folds for the first configuration, then all requested folds for the next configuration.
- Defaults are tailored for this fork: `--plan nnUNetCondUNetPlans` and `--trainer nnUNetTrainerAdamW`.
- `--disable-tta` disables validation test-time augmentation for all scheduled jobs; TTA remains enabled by default when the flag is omitted.

## Trainer Lookup

- `nnunetv2/utilities/find_objects.py` and `nnunetv2/utilities/find_class_by_name.py` include local behavior around external trainer lookup via `nnUNet_extTrainer`.
- Tests for this behavior are in `nnunetv2/tests/test_find_objects.py`.

# Useful Verification Commands

- Sync environment: `uv sync --extra dev`
- Focused planner/config tests: `uv run pytest nnunetv2/tests/test_cond_unet_planner_and_config.py`
- Full local test suite: `uv run pytest nnunetv2/tests`
- Compile touched Python files: `uv run python -m compileall -q <paths>`

Keep `uv.lock` updated when dependency metadata changes.
