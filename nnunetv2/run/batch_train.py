from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from os.path import abspath, dirname, isfile, join
from typing import Callable, Sequence

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.run_training import maybe_load_checkpoint
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


@dataclass(frozen=True)
class TrainingJob:
    index: int
    total: int
    configuration: str
    fold: int


@dataclass
class RunningJob:
    job: TrainingJob
    gpu: str
    process: subprocess.Popen
    start_time: float


@dataclass(frozen=True)
class FinishedJob:
    job: TrainingJob
    gpu: str
    returncode: int
    elapsed_seconds: float


JobPair = tuple[str, int]


def format_duration(seconds: float) -> str:
    minutes = max(0, int(seconds // 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def build_jobs(configurations: Sequence[str], folds: Sequence[int]) -> list[TrainingJob]:
    return build_jobs_from_pairs([(configuration, fold) for configuration in configurations for fold in folds])


def build_jobs_from_pairs(job_pairs: Sequence[JobPair]) -> list[TrainingJob]:
    total = len(job_pairs)
    jobs = []
    for configuration, fold in job_pairs:
        jobs.append(TrainingJob(len(jobs), total, configuration, fold))
    return jobs


def apply_job_overrides(jobs: Sequence[TrainingJob],
                        include: Sequence[JobPair] | None = None,
                        exclude: Sequence[JobPair] | None = None) -> list[TrainingJob]:
    include = include or ()
    exclude = exclude or ()
    excluded = set(exclude)

    job_pairs: list[JobPair] = [(job.configuration, job.fold) for job in jobs]
    seen: set[JobPair] = set(job_pairs)

    for pair in include:
        if pair not in seen:
            job_pairs.append(pair)
            seen.add(pair)

    return build_jobs_from_pairs([pair for pair in job_pairs if pair not in excluded])


def requested_configurations(configurations: Sequence[str],
                             include: Sequence[JobPair] | None = None,
                             exclude: Sequence[JobPair] | None = None) -> list[str]:
    requested = []
    seen = set()
    for configuration in list(configurations) + [i[0] for i in include or ()] + [i[0] for i in exclude or ()]:
        if configuration not in seen:
            requested.append(configuration)
            seen.add(configuration)
    return requested


def visible_gpu_tokens(cuda_visible_devices: str | None, device_count: int) -> list[str]:
    if device_count < 1:
        return []
    if cuda_visible_devices:
        tokens = [i.strip() for i in cuda_visible_devices.split(",") if i.strip()]
        if len(tokens) >= device_count:
            return tokens[:device_count]
    return [str(i) for i in range(device_count)]


def parse_job_pair(value: str) -> JobPair:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"invalid job pair {value!r}; expected CONFIG,FOLD, for example 3x-s,0"
        )

    configuration = parts[0].strip()
    fold_text = parts[1].strip()
    if not configuration or not fold_text:
        raise argparse.ArgumentTypeError(
            f"invalid job pair {value!r}; expected CONFIG,FOLD, for example 3x-s,0"
        )

    try:
        fold = int(fold_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid fold in job pair {value!r}; expected an integer fold"
        ) from exc

    return configuration, fold


def resolve_plans_file(dataset_name_or_id: str, plans: str) -> str:
    if isfile(plans):
        return abspath(plans)

    if plans.endswith(".json") or os.sep in plans:
        raise FileNotFoundError(f"Plans file does not exist: {plans}")

    if not nnUNet_preprocessed.is_set():
        raise RuntimeError(
            "nnUNet_preprocessed is not set. Pass a plans JSON path with -p/--plan or set nnUNet_preprocessed "
            "so the batch trainer can discover plans by identifier."
        )

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    candidate = join(nnUNet_preprocessed.require(), dataset_name, plans + ".json")
    if not isfile(candidate):
        raise FileNotFoundError(f"Could not find plans file: {candidate}")
    return abspath(candidate)


def validate_requested_configurations(plans_file: str, configurations: Sequence[str]) -> None:
    plans_manager = PlansManager(plans_file)
    missing = [i for i in configurations if i not in plans_manager.available_configurations]
    if missing:
        raise RuntimeError(
            f"Requested configuration(s) not found in {plans_file}: {missing}. "
            f"Available configurations: {plans_manager.available_configurations}"
        )


def validate_plans_dataset(plans_file: str, dataset_name_or_id: str) -> None:
    plans_manager = PlansManager(plans_file)
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    if plans_manager.dataset_name != dataset_name:
        raise RuntimeError(
            f"Plans file {plans_file} belongs to {plans_manager.dataset_name}, "
            f"but the requested dataset is {dataset_name}."
        )


def load_dataset_json_for_plans(plans_file: str) -> dict:
    dataset_json_file = join(dirname(plans_file), "dataset.json")
    if not isfile(dataset_json_file):
        raise FileNotFoundError(f"Could not find dataset.json next to plans file: {dataset_json_file}")
    return load_json(dataset_json_file)


def run_training_worker(plans_file: str,
                        configuration: str,
                        fold: int,
                        trainer_name: str,
                        disable_tta: bool) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    plans = load_json(plans_file)
    plans["continue_training"] = False
    dataset_json = load_dataset_json_for_plans(plans_file)
    trainer_class = recursive_find_trainer_class_by_name(trainer_name)
    trainer = trainer_class(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=torch.device("cuda"),
    )

    maybe_load_checkpoint(trainer, continue_training=False, validation_only=False, pretrained_weights_file=None)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    trainer.run_training()
    trainer.perform_actual_validation(False, not disable_tta)


def make_worker_command(plans_file: str,
                        configuration: str,
                        fold: int,
                        trainer_name: str,
                        disable_tta: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "nnunetv2.run.batch_train",
        "--worker",
        "--plans-file",
        plans_file,
        "--configuration",
        configuration,
        "--fold",
        str(fold),
        "--trainer",
        trainer_name,
    ]
    if disable_tta:
        command.append("--disable-tta")
    return command


def launch_job(job: TrainingJob,
               gpu: str,
               plans_file: str,
               trainer_name: str,
               disable_tta: bool) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    return subprocess.Popen(
        make_worker_command(plans_file, job.configuration, job.fold, trainer_name, disable_tta),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def log_start(console: Console, job: TrainingJob, gpu: str) -> None:
    console.print(
        Panel(
            f"[bold cyan]START[/bold cyan] [bold]job {job.index}/{job.total - 1}[/bold] "
            f"on GPU [bold]{gpu}[/bold]\n"
            f"config=[bold]{job.configuration}[/bold] fold=[bold]{job.fold}[/bold]",
            border_style="cyan",
        )
    )


def log_finish(console: Console, result: FinishedJob) -> None:
    if result.returncode == 0:
        status = "[bold green]DONE[/bold green]"
        border_style = "green"
    else:
        status = f"[bold red]FAILED[/bold red] returncode=[bold]{result.returncode}[/bold]"
        border_style = "red"
    console.print(
        Panel(
            f"{status} [bold]job {result.job.index}/{result.job.total - 1}[/bold] "
            f"on GPU [bold]{result.gpu}[/bold]\n"
            f"config=[bold]{result.job.configuration}[/bold] fold=[bold]{result.job.fold}[/bold] "
            f"time=[bold]{format_duration(result.elapsed_seconds)}[/bold]",
            border_style=border_style,
        )
    )


def print_summary(console: Console, results: Sequence[FinishedJob]) -> None:
    failures = [i for i in results if i.returncode != 0]
    table = Table(title="Batch training summary")
    table.add_column("Total", justify="right")
    table.add_column("Succeeded", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Failed jobs", style="red")
    table.add_row(
        str(len(results)),
        str(len(results) - len(failures)),
        str(len(failures)),
        ", ".join(str(i.job.index) for i in failures) if failures else "-",
    )
    console.print(table)


def schedule_jobs(jobs: Sequence[TrainingJob],
                  gpu_tokens: Sequence[str],
                  plans_file: str,
                  trainer_name: str,
                  disable_tta: bool,
                  console: Console,
                  launcher: Callable[[TrainingJob, str, str, str, bool], subprocess.Popen] = launch_job,
                  monotonic: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep,
                  poll_interval: float = 5.0) -> list[FinishedJob]:
    pending = list(jobs)
    free_gpus = list(gpu_tokens)
    running: list[RunningJob] = []
    results: list[FinishedJob] = []

    while pending or running:
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            job = pending.pop(0)
            process = launcher(job, gpu, plans_file, trainer_name, disable_tta)
            running.append(RunningJob(job, gpu, process, monotonic()))
            log_start(console, job, gpu)

        finished = []
        for running_job in running:
            returncode = running_job.process.poll()
            if returncode is not None:
                result = FinishedJob(
                    running_job.job,
                    running_job.gpu,
                    returncode,
                    monotonic() - running_job.start_time,
                )
                results.append(result)
                finished.append(running_job)
                free_gpus.append(running_job.gpu)
                log_finish(console, result)

        if finished:
            running = [i for i in running if i not in finished]
        elif running:
            sleep(poll_interval)

    return results


def get_visible_gpus() -> list[str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nnUNetv2_batch_train, but torch.cuda.is_available() is False.")
    return visible_gpu_tokens(os.environ.get("CUDA_VISIBLE_DEVICES"), torch.cuda.device_count())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schedule multiple single-GPU nnU-Net training jobs.")
    parser.add_argument("-d", "--dataset", help="Dataset ID or DatasetXXX_name.")
    parser.add_argument(
        "-t",
        "--trainer",
        default="nnUNetTrainerAdamW",
        help="Trainer class name. Default: nnUNetTrainerAdamW",
    )
    parser.add_argument(
        "-p",
        "--plan",
        default="nnUNetCondUNetPlans",
        help="Plans identifier under nnUNet_preprocessed/<dataset>, or a direct plans JSON path. "
             "Default: nnUNetCondUNetPlans",
    )
    parser.add_argument("-c", "--configs", nargs="+", help="Configuration names.")
    parser.add_argument("-f", "--folds", nargs="+", type=int, help="Fold indices.")
    parser.add_argument(
        "-i",
        "--include",
        nargs="+",
        type=parse_job_pair,
        default=(),
        metavar="CONFIG,FOLD",
        help="Additional individual jobs to schedule, for example: -i 3x-s,0 4x-m,1",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        nargs="+",
        type=parse_job_pair,
        default=(),
        metavar="CONFIG,FOLD",
        help="Individual jobs to remove from the schedule, for example: -e 3x-s,0 4x-m,1",
    )
    parser.add_argument(
        "--disable-tta",
        action="store_true",
        default=False,
        help="Disable test-time augmentation during post-training validation.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plans-file", help=argparse.SUPPRESS)
    parser.add_argument("--configuration", help=argparse.SUPPRESS)
    parser.add_argument("--fold", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        missing = [i for i in ("plans_file", "configuration", "fold", "trainer") if getattr(args, i) is None]
    else:
        missing = [i for i in ("dataset", "configs", "folds") if getattr(args, i) is None]
    if missing:
        parser.error("missing required argument(s): " + ", ".join("--" + i.replace("_", "-") for i in missing))
    return args


def batch_train_entry(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.worker:
        run_training_worker(args.plans_file, args.configuration, args.fold, args.trainer, args.disable_tta)
        return 0

    console = Console()
    plans_file = resolve_plans_file(args.dataset, args.plan)
    validate_plans_dataset(plans_file, args.dataset)
    validate_requested_configurations(
        plans_file,
        requested_configurations(args.configs, args.include, args.exclude),
    )

    gpu_tokens = get_visible_gpus()
    if not gpu_tokens:
        raise RuntimeError("No visible CUDA GPUs found.")

    jobs = apply_job_overrides(build_jobs(args.configs, args.folds), args.include, args.exclude)
    console.print(
        f"[bold]Scheduling {len(jobs)} jobs[/bold] across [bold]{len(gpu_tokens)} GPUs[/bold] "
        f"with plans [bold]{plans_file}[/bold]"
    )
    results = schedule_jobs(jobs, gpu_tokens, plans_file, args.trainer, args.disable_tta, console)
    print_summary(console, results)
    return 1 if any(i.returncode != 0 for i in results) else 0


def main() -> None:
    raise SystemExit(batch_train_entry())


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    main()
