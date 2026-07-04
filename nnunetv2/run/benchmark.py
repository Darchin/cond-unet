from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
from os.path import abspath, dirname, isfile, join
from statistics import mean, stdev
from typing import Iterable

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from torch import autocast

try:
    from torch import GradScaler
    TORCH_HAS_OLD_GRADSCALER = False
except ImportError:
    from torch.cuda.amp import GradScaler
    TORCH_HAS_OLD_GRADSCALER = True

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class BenchmarkOOM(RuntimeError):
    pass


def _is_oom_error(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _clear_cuda_after_oom() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _matching_plan_files(plans_identifier: str, configuration: str) -> list[str]:
    if not nnUNet_preprocessed.is_set():
        raise RuntimeError(
            "nnUNet_preprocessed is not set. Pass a plans JSON path with -pl or set nnUNet_preprocessed "
            "so the benchmark can discover plans by identifier."
        )

    candidates = sorted(glob.glob(join(nnUNet_preprocessed.require(), "*", plans_identifier + ".json")))
    matches = []
    for candidate in candidates:
        try:
            plans = load_json(candidate)
        except Exception:
            continue
        if configuration in plans.get("configurations", {}):
            matches.append(candidate)
    return matches


def resolve_plans_file(plans: str, configuration: str) -> str:
    if isfile(plans):
        return abspath(plans)

    if plans.endswith(".json") or os.sep in plans:
        raise FileNotFoundError(f"Plans file does not exist: {plans}")

    matches = _matching_plan_files(plans, configuration)
    if not matches:
        raise FileNotFoundError(
            f"Could not find a {plans}.json file containing configuration '{configuration}' under "
            f"nnUNet_preprocessed={nnUNet_preprocessed.get()!r}."
        )
    if len(matches) > 1:
        formatted = "\n".join(f"  - {i}" for i in matches)
        raise RuntimeError(
            f"Configuration '{configuration}' with plans identifier '{plans}' is ambiguous. "
            f"Pass one of these files directly via -pl:\n{formatted}"
        )
    return matches[0]


def load_dataset_json_for_plans(plans_file: str) -> dict:
    dataset_json_file = join(dirname(plans_file), "dataset.json")
    if not isfile(dataset_json_file):
        raise FileNotFoundError(
            f"Could not find dataset.json next to plans file: {dataset_json_file}. "
            "The benchmark needs it to determine input channels and output heads."
        )
    return load_json(dataset_json_file)


def build_dice_ce_loss(configuration_manager: ConfigurationManager,
                       plans_manager: PlansManager,
                       dataset_json: dict,
                       enable_deep_supervision: bool) -> torch.nn.Module:
    label_manager = plans_manager.get_label_manager(dataset_json)
    loss = DC_and_CE_loss(
        {
            "batch_dice": configuration_manager.batch_dice,
            "smooth": 1e-5,
            "do_bg": False,
            "ddp": False,
        },
        {},
        weight_ce=1,
        weight_dice=1,
        ignore_label=label_manager.ignore_label,
        dice_class=MemoryEfficientSoftDiceLoss,
    )

    if enable_deep_supervision:
        deep_supervision_scales = list(
            list(i) for i in 1 / np.cumprod(np.vstack(configuration_manager.pool_op_kernel_sizes), axis=0)
        )[:-1]
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0
        weights = weights / weights.sum()
        loss = DeepSupervisionWrapper(loss, weights)
    return loss


def _iter_outputs(output: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> Iterable[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return (output,)
    return output


def make_targets_like(output: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
                      num_classes: int,
                      device: torch.device) -> torch.Tensor | list[torch.Tensor]:
    targets = [
        torch.randint(0, num_classes, (i.shape[0], 1, *i.shape[2:]), device=device, dtype=torch.int16)
        for i in _iter_outputs(output)
    ]
    return targets[0] if isinstance(output, torch.Tensor) else targets


def run_step(network: torch.nn.Module,
             loss_fn: torch.nn.Module,
             data: torch.Tensor,
             target: torch.Tensor | list[torch.Tensor],
             grad_scaler: GradScaler | None,
             measure: bool) -> tuple[float, float, float]:
    network.zero_grad(set_to_none=True)

    if not measure:
        with autocast("cuda", enabled=True):
            output = network(data)
            loss = loss_fn(output, target)
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.cuda.synchronize()
        return 0.0, 0.0, 0.0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)

    forward_start.record()
    with autocast("cuda", enabled=True):
        output = network(data)
        loss = loss_fn(output, target)
    forward_end.record()

    backward_start.record()
    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
    else:
        loss.backward()
    backward_end.record()
    torch.cuda.synchronize()

    forward_time = forward_start.elapsed_time(forward_end)
    backward_time = backward_start.elapsed_time(backward_end)
    peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
    return peak_memory, forward_time, backward_time


def format_mean_std(values: list[float], precision: int = 3) -> str:
    value_mean = mean(values)
    value_std = stdev(values) if len(values) > 1 else 0.0
    return f"{value_mean:.{precision}f} +/- {value_std:.{precision}f}"


def print_results(memory_gb: list[float], forward_ms: list[float], backward_ms: list[float]) -> None:
    total_ms = [f + b for f, b in zip(forward_ms, backward_ms)]
    rows = [
        ("Peak memory per full step (GB)", format_mean_std(memory_gb, 3)),
        ("Forward time per step (ms)", format_mean_std(forward_ms, 3)),
        ("Backward time per step (ms)", format_mean_std(backward_ms, 3)),
        ("Total time per step (ms)", format_mean_std(total_ms, 3)),
    ]
    metric_width = max(len("Metric"), *(len(i[0]) for i in rows))
    value_width = max(len("Mean +/- std dev"), *(len(i[1]) for i in rows))
    separator = f"+-{'-' * metric_width}-+-{'-' * value_width}-+"
    print(separator)
    print(f"| {'Metric'.ljust(metric_width)} | {'Mean +/- std dev'.ljust(value_width)} |")
    print(separator)
    for metric, value in rows:
        print(f"| {metric.ljust(metric_width)} | {value.rjust(value_width)} |")
    print(separator)


def benchmark(configuration: str,
              plans_arg: str,
              num_warmup: int,
              num_repeats: int,
              compile_model: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nnUNetv2_benchmark, but torch.cuda.is_available() is False.")
    if num_warmup < 0:
        raise ValueError("-nw must be >= 0")
    if num_repeats < 1:
        raise ValueError("-nr must be >= 1")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    plans_file = resolve_plans_file(plans_arg, configuration)
    plans_manager = PlansManager(plans_file)
    configuration_manager = plans_manager.get_configuration(configuration)
    configuration_manager.validate_required_for_training(configuration)
    dataset_json = load_dataset_json_for_plans(plans_file)

    label_manager = plans_manager.get_label_manager(dataset_json)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    num_output_channels = label_manager.num_segmentation_heads
    enable_deep_supervision = configuration_manager.trainer.get("enable_deep_supervision", True)

    print(f"Plans file: {plans_file}")
    print(f"Configuration: {configuration}")
    print(f"Input shape: {[configuration_manager.batch_size, num_input_channels, *configuration_manager.patch_size]}")
    print(f"Output channels: {num_output_channels}")
    print(f"Deep supervision: {enable_deep_supervision}")
    print(f"torch.compile: {compile_model}")

    try:
        network = nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        ).to(device)
        network.train()
        network = torch.compile(network, dynamic=False, disable=not compile_model)
        loss_fn = build_dice_ce_loss(
            configuration_manager, plans_manager, dataset_json, enable_deep_supervision
        ).to(device)

        data = torch.randn(
            (configuration_manager.batch_size, num_input_channels, *configuration_manager.patch_size),
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad(), autocast("cuda", enabled=True):
            sample_output = network(data)
        target = make_targets_like(sample_output, num_output_channels, device)
        del sample_output

        grad_scaler = GradScaler("cuda") if not TORCH_HAS_OLD_GRADSCALER else GradScaler()

        for _ in range(num_warmup):
            run_step(network, loss_fn, data, target, grad_scaler, measure=False)

        memory_gb, forward_ms, backward_ms = [], [], []
        for _ in range(num_repeats):
            memory, forward, backward = run_step(network, loss_fn, data, target, grad_scaler, measure=True)
            memory_gb.append(memory)
            forward_ms.append(forward)
            backward_ms.append(backward)

        print_results(memory_gb, forward_ms, backward_ms)
    except BaseException as error:
        if _is_oom_error(error):
            _clear_cuda_after_oom()
            raise BenchmarkOOM(
                "CUDA ran out of memory while benchmarking this configuration. "
                "Try a smaller batch size or patch size, or run without torch.compile."
            ) from error
        raise


def benchmark_entry() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA memory and forward/backward time for an nnU-Net v2 configuration."
    )
    parser.add_argument("configuration", type=str, help="Configuration name from the plans file.")
    parser.add_argument(
        "-pl",
        type=str,
        default="nnUNetCondUNetPlans",
        help="Plans identifier to discover under nnUNet_preprocessed, or a direct path to a plans JSON file. "
             "Default: nnUNetCondUNetPlans",
    )
    parser.add_argument("-nw", type=int, default=10, help="Number of warm-up steps. Default: 10")
    parser.add_argument(
        "-nr",
        type=int,
        default=25,
        help="Number of measured repeats after warm-up. Default: 25",
    )
    parser.add_argument(
        "-c",
        action="store_true",
        help="Enable torch.compile(dynamic=False). The flag is passed through torch.compile's disable parameter.",
    )
    args = parser.parse_args()

    try:
        benchmark(args.configuration, args.pl, args.nw, args.nr, args.c)
    except BenchmarkOOM as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    benchmark_entry()
