from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from os.path import abspath, dirname, isabs, isfile, join
from typing import Callable, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import load_json
from rich.console import Console
from rich.table import Table

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.run_training import find_free_network_port, maybe_load_checkpoint
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
    started_at: datetime


@dataclass(frozen=True)
class FinishedJob:
    job: TrainingJob
    gpu: str
    returncode: int
    elapsed_seconds: float
    started_at: datetime
    finished_at: datetime


JobPair = tuple[str, int]
PROJECT_ROOT = abspath(join(dirname(__file__), "..", ".."))
JOBS_DIR = join(PROJECT_ROOT, "jobs")
STARTED_COLOR = "deep_sky_blue1"
FINISHED_COLOR = "chartreuse3"
FAILED_COLOR = "red3"


def format_duration(seconds: float) -> str:
    minutes = max(0, int(seconds // 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def format_bold_value(value: object, color: str | None = None) -> str:
    if color is None:
        return f"[bold]{value}[/bold]"
    return f"[bold {color}]{value}[/bold {color}]"


def format_verbose_duration(seconds: float, color: str | None = None) -> str:
    minutes = max(0, int(seconds // 60))
    hours, minutes = divmod(minutes, 60)
    hour_label = "hour" if hours == 1 else "hours"
    minute_label = "minute" if minutes == 1 else "minutes"
    if color is None:
        return f"{format_bold_value(hours)} {hour_label} and {format_bold_value(minutes)} {minute_label}"
    duration = f"{hours} {hour_label} and {minutes} {minute_label}"
    return format_bold_value(duration, color)


def format_log_timestamp(timestamp: datetime, color: str | None = None) -> str:
    return format_bold_value(timestamp.strftime("%Y-%m-%d at %H:%M"), color)


def format_job_progress(job: TrainingJob, gpu: str, color: str | None = None) -> str:
    return f"job {format_bold_value(f'{job.index + 1}/{job.total}', color)} on GPU {format_bold_value(gpu, color)}"


def format_job_configuration(job: TrainingJob, color: str | None = None) -> str:
    return f"fold {format_bold_value(job.fold, color)} of config {format_bold_value(job.configuration, color)}"


def format_start_message(job: TrainingJob, gpu: str, started_at: datetime) -> str:
    return (
        f"{format_bold_value('STARTED', STARTED_COLOR)} {format_job_progress(job, gpu, STARTED_COLOR)} — "
        f"{format_job_configuration(job, STARTED_COLOR)} — "
        f"started on {format_log_timestamp(started_at, STARTED_COLOR)}."
    )


def format_finish_message(result: FinishedJob) -> str:
    color = FINISHED_COLOR if result.returncode == 0 else FAILED_COLOR
    status = "FINISHED" if result.returncode == 0 else "FAILED"
    return (
        f"{format_bold_value(status, color)} {format_job_progress(result.job, result.gpu, color)} — "
        f"{format_job_configuration(result.job, color)} — "
        f"started on {format_log_timestamp(result.started_at, color)}, "
        f"finished on {format_log_timestamp(result.finished_at, color)}, "
        f"took {format_verbose_duration(result.elapsed_seconds, color)} in total."
    )


def build_jobs(configurations: Sequence[str], folds: Sequence[int]) -> list[TrainingJob]:
    return build_jobs_from_pairs([(configuration, fold) for configuration in configurations for fold in folds])


def build_jobs_from_pairs(job_pairs: Sequence[JobPair]) -> list[TrainingJob]:
    total = len(job_pairs)
    jobs = []
    for configuration, fold in job_pairs:
        jobs.append(TrainingJob(len(jobs), total, configuration, fold))
    return jobs


def format_job_pair(pair: JobPair) -> str:
    return f"{pair[0]},{pair[1]}"


def apply_job_overrides(jobs: Sequence[TrainingJob],
                        include: Sequence[JobPair] | None = None,
                        exclude: Sequence[JobPair] | None = None) -> list[TrainingJob]:
    original_pairs = [(job.configuration, job.fold) for job in jobs]
    return build_jobs_from_pairs(resolve_job_overrides(original_pairs, include, exclude))


def resolve_job_overrides(original_pairs: Sequence[JobPair],
                          include: Sequence[JobPair] | None = None,
                          exclude: Sequence[JobPair] | None = None) -> list[JobPair]:
    include = include or ()
    exclude = exclude or ()
    original_pair_set = set(original_pairs)
    exclude_set = set(exclude)

    include_exclude_overlap = [pair for pair in include if pair in exclude_set]
    if include_exclude_overlap:
        raise ValueError(
            "Job pair(s) cannot appear in both include and exclude: "
            + ", ".join(format_job_pair(pair) for pair in include_exclude_overlap)
        )

    included_existing = [pair for pair in include if pair in original_pair_set]
    if included_existing:
        raise ValueError(
            "Included job pair(s) already exist in the original configuration/fold combinations: "
            + ", ".join(format_job_pair(pair) for pair in included_existing)
        )

    excluded_missing = [pair for pair in exclude if pair not in original_pair_set]
    if excluded_missing:
        raise ValueError(
            "Excluded job pair(s) do not exist in the original configuration/fold combinations: "
            + ", ".join(format_job_pair(pair) for pair in excluded_missing)
        )

    resolved = [pair for pair in original_pairs if pair not in exclude_set]
    seen_includes: set[JobPair] = set()
    for pair in include:
        if pair in seen_includes:
            continue
        resolved.append(pair)
        seen_includes.add(pair)
    return resolved


def resolve_cli_job_pairs(configurations: Sequence[str] | None,
                          folds: Sequence[int] | None,
                          include: Sequence[JobPair] | None = None,
                          exclude: Sequence[JobPair] | None = None) -> list[JobPair]:
    if (configurations is None) != (folds is None):
        raise ValueError("Configuration names and fold indices must be provided together.")

    original_pairs = [] if configurations is None else [
        (configuration, fold) for configuration in configurations for fold in folds or ()
    ]
    job_pairs = resolve_job_overrides(original_pairs, include, exclude)
    if not job_pairs:
        raise ValueError("Resolved job list is empty.")
    return job_pairs


def validate_json_array(value: object,
                        *,
                        key: str,
                        entry_index: int,
                        expected_type: type) -> list:
    expected_name = expected_type.__name__
    if not isinstance(value, list):
        raise ValueError(
            f"Job JSON entry {entry_index} field {key!r} must be a "
            f"list of {expected_name} values."
        )
    if not value:
        raise ValueError(f"Job JSON entry {entry_index} field {key!r} must not be empty.")
    invalid = [i for i in value if type(i) is not expected_type]
    if invalid:
        raise ValueError(
            f"Job JSON entry {entry_index} field {key!r} contains invalid value(s): {invalid!r}"
        )
    if expected_type is str:
        empty = [i for i in value if not i]
        if empty:
            raise ValueError(f"Job JSON entry {entry_index} field {key!r} must not contain empty strings.")
    return value


def load_job_pairs_from_json(json_file: str) -> list[JobPair]:
    json_file = resolve_job_json_file(json_file)
    job_specs = load_json(json_file)
    if not isinstance(job_specs, list):
        raise ValueError("Job JSON must contain a top-level array.")
    if not job_specs:
        raise ValueError("Job JSON must contain at least one job entry.")

    job_pairs: list[JobPair] = []
    seen_by_entry: dict[JobPair, int] = {}
    for entry_index, entry in enumerate(job_specs):
        if not isinstance(entry, dict):
            raise ValueError(f"Job JSON entry {entry_index} must be an object.")
        extra_keys = sorted(set(entry) - {"configs", "folds"})
        if extra_keys:
            raise ValueError(f"Job JSON entry {entry_index} contains unsupported key(s): {extra_keys}")
        if "configs" not in entry or "folds" not in entry:
            raise ValueError(f"Job JSON entry {entry_index} must contain 'configs' and 'folds'.")

        configurations = validate_json_array(
            entry["configs"],
            key="configs",
            entry_index=entry_index,
            expected_type=str,
        )
        folds = validate_json_array(
            entry["folds"],
            key="folds",
            entry_index=entry_index,
            expected_type=int,
        )

        entry_pairs: list[JobPair] = [(configuration, fold) for configuration in configurations for fold in folds]
        entry_seen: set[JobPair] = set()
        entry_duplicates = []
        for pair in entry_pairs:
            if pair in entry_seen:
                entry_duplicates.append(pair)
            entry_seen.add(pair)
        if entry_duplicates:
            raise ValueError(
                f"Job JSON entry {entry_index} contains duplicate job pair(s): "
                + ", ".join(format_job_pair(pair) for pair in entry_duplicates)
            )

        overlapping_pairs = [pair for pair in entry_pairs if pair in seen_by_entry]
        if overlapping_pairs:
            raise ValueError(
                f"Job JSON entry {entry_index} overlaps earlier entries: "
                + ", ".join(format_job_pair(pair) for pair in overlapping_pairs)
            )

        for pair in entry_pairs:
            seen_by_entry[pair] = entry_index
            job_pairs.append(pair)

    return job_pairs


def resolve_job_json_file(json_file: str) -> str:
    if not json_file.endswith(".json"):
        jobs_candidate = abspath(join(JOBS_DIR, json_file + ".json"))
        if isfile(jobs_candidate):
            return jobs_candidate
        raise FileNotFoundError(f"Job JSON file does not exist in {JOBS_DIR}: {json_file}.json")

    direct_candidate = json_file if isabs(json_file) else abspath(json_file)
    if isfile(direct_candidate):
        return direct_candidate
    raise FileNotFoundError(f"Job JSON file does not exist: {direct_candidate}")


def resolve_jobs(args: argparse.Namespace) -> list[TrainingJob]:
    if args.json:
        job_pairs = load_job_pairs_from_json(args.json)
    else:
        job_pairs = resolve_cli_job_pairs(args.configs, args.folds, args.include, args.exclude)
    return build_jobs_from_pairs(job_pairs)


def requested_configurations_from_jobs(jobs: Sequence[TrainingJob],
                                       extra_pairs: Sequence[JobPair] | None = None) -> list[str]:
    requested = []
    seen = set()
    for configuration in [job.configuration for job in jobs] + [pair[0] for pair in extra_pairs or ()]:
        if configuration not in seen:
            requested.append(configuration)
            seen.add(configuration)
    return requested


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


def group_gpu_tokens(gpu_tokens: Sequence[str], ddp: int) -> list[str]:
    return [",".join(gpu_tokens[i:i + ddp]) for i in range(0, len(gpu_tokens) - ddp + 1, ddp)]


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


def validate_ddp_batch_sizes(plans_file: str, configurations: Sequence[str], ddp: int) -> None:
    plans_manager = PlansManager(plans_file)
    invalid = []
    for configuration in configurations:
        batch_size = plans_manager.get_configuration(configuration).batch_size
        if batch_size % ddp != 0:
            invalid.append(f"{configuration} (batch size {batch_size})")
    if invalid:
        raise RuntimeError(
            f"--ddp {ddp} must divide the batch size of every requested configuration; incompatible: "
            + ", ".join(invalid)
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


def run_training_worker_process(rank: int,
                                world_size: int,
                                plans_file: str,
                                configuration: str,
                                fold: int,
                                trainer_name: str,
                                disable_tta: bool,
                                checkpoint_interval: int,
                                disable_train_val: bool) -> None:
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

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
        device=torch.device("cuda", rank) if world_size > 1 else torch.device("cuda"),
    )
    trainer.checkpoint_interval = checkpoint_interval
    trainer.disable_train_val = disable_train_val

    maybe_load_checkpoint(trainer, continue_training=False, validation_only=False, pretrained_weights_file=None)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    trainer.run_training()
    trainer.perform_actual_validation(False, not disable_tta)

    if world_size > 1:
        dist.destroy_process_group()


def run_training_worker(plans_file: str,
                        configuration: str,
                        fold: int,
                        trainer_name: str,
                        disable_tta: bool,
                        checkpoint_interval: int,
                        disable_train_val: bool,
                        ddp: int) -> None:
    if ddp == 1:
        run_training_worker_process(
            0, 1, plans_file, configuration, fold, trainer_name, disable_tta,
            checkpoint_interval, disable_train_val,
        )
        return

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_network_port())
    mp.spawn(
        run_training_worker_process,
        args=(
            ddp, plans_file, configuration, fold, trainer_name, disable_tta,
            checkpoint_interval, disable_train_val,
        ),
        nprocs=ddp,
        join=True,
    )


def make_worker_command(plans_file: str,
                        configuration: str,
                        fold: int,
                        trainer_name: str,
                        disable_tta: bool,
                        checkpoint_interval: int,
                        disable_train_val: bool,
                        ddp: int = 1) -> list[str]:
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
        "--ckpt-interval",
        str(checkpoint_interval),
        "--ddp",
        str(ddp),
    ]
    if disable_tta:
        command.append("--disable-tta")
    if disable_train_val:
        command.append("--disable_train_val")
    return command


def launch_job(job: TrainingJob,
               gpu: str,
               plans_file: str,
               trainer_name: str,
               disable_tta: bool,
               checkpoint_interval: int,
               disable_train_val: bool,
               ddp: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    return subprocess.Popen(
        make_worker_command(plans_file, job.configuration, job.fold, trainer_name, disable_tta,
                            checkpoint_interval, disable_train_val, ddp),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def log_start(console: Console, job: TrainingJob, gpu: str, started_at: datetime) -> None:
    console.print(format_start_message(job, gpu, started_at))


def log_finish(console: Console, result: FinishedJob) -> None:
    console.print(format_finish_message(result))


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
                  checkpoint_interval: int,
                  disable_train_val: bool,
                  console: Console,
                  ddp: int = 1,
                  launcher: Callable[[TrainingJob, str, str, str, bool, int, bool, int], subprocess.Popen] = launch_job,
                  monotonic: Callable[[], float] = time.monotonic,
                  wall_clock: Callable[[], datetime] = datetime.now,
                  sleep: Callable[[float], None] = time.sleep,
                  poll_interval: float = 5.0) -> list[FinishedJob]:
    pending = list(jobs)
    free_gpus = group_gpu_tokens(gpu_tokens, ddp)
    if pending and not free_gpus:
        raise ValueError(f"Cannot schedule jobs with --ddp {ddp} across {len(gpu_tokens)} GPU(s).")
    running: list[RunningJob] = []
    results: list[FinishedJob] = []

    while pending or running:
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            job = pending.pop(0)
            process = launcher(job, gpu, plans_file, trainer_name, disable_tta,
                               checkpoint_interval, disable_train_val, ddp)
            started_at = wall_clock()
            running.append(RunningJob(job, gpu, process, monotonic(), started_at))
            log_start(console, job, gpu, started_at)

        finished = []
        for running_job in running:
            returncode = running_job.process.poll()
            if returncode is not None:
                finished_at = wall_clock()
                result = FinishedJob(
                    running_job.job,
                    running_job.gpu,
                    returncode,
                    monotonic() - running_job.start_time,
                    running_job.started_at,
                    finished_at,
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
    parser = argparse.ArgumentParser(description="Schedule multiple nnU-Net training jobs across GPUs.")
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
    parser.add_argument("-j", "--json", help="Path to a JSON file describing ordered training jobs.")
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
        "--ddp",
        type=int,
        default=1,
        help="Number of GPUs assigned to each job. Default: 1.",
    )
    parser.add_argument(
        "--disable-tta",
        action="store_true",
        default=False,
        help="Disable test-time augmentation during post-training validation.",
    )
    parser.add_argument(
        "--ckpt-interval",
        type=int,
        default=50,
        help="Save checkpoint_last.pth after this many epochs. Default: 50.",
    )
    parser.add_argument(
        "--disable_train_val",
        action="store_true",
        default=False,
        help="Disable validation loops and pseudo-Dice computation during training.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plans-file", help=argparse.SUPPRESS)
    parser.add_argument("--configuration", help=argparse.SUPPRESS)
    parser.add_argument("--fold", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.ckpt_interval <= 0:
        parser.error("--ckpt-interval must be greater than 0")
    if args.ddp <= 0:
        parser.error("--ddp must be greater than 0")

    if args.worker:
        missing = [i for i in ("plans_file", "configuration", "fold", "trainer") if getattr(args, i) is None]
    else:
        missing = [i for i in ("dataset",) if getattr(args, i) is None]
    if missing:
        parser.error("missing required argument(s): " + ", ".join("--" + i.replace("_", "-") for i in missing))
    if not args.worker and args.json and any((args.configs, args.folds, args.include, args.exclude)):
        parser.error("--json cannot be combined with --configs, --folds, --include, or --exclude")
    return args


def batch_train_entry(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.worker:
        run_training_worker(args.plans_file, args.configuration, args.fold, args.trainer, args.disable_tta,
                            args.ckpt_interval, args.disable_train_val, args.ddp)
        return 0

    console = Console()
    jobs = resolve_jobs(args)
    plans_file = resolve_plans_file(args.dataset, args.plan)
    validate_plans_dataset(plans_file, args.dataset)
    validate_requested_configurations(
        plans_file,
        requested_configurations_from_jobs(jobs, args.exclude),
    )
    validate_ddp_batch_sizes(plans_file, requested_configurations_from_jobs(jobs), args.ddp)

    gpu_tokens = get_visible_gpus()
    if not gpu_tokens:
        raise RuntimeError("No visible CUDA GPUs found.")
    if len(gpu_tokens) < args.ddp:
        raise RuntimeError(f"--ddp {args.ddp} requires at least {args.ddp} visible GPUs, found {len(gpu_tokens)}.")

    console.print(
        f"[bold]Scheduling {len(jobs)} jobs[/bold] across [bold]{len(gpu_tokens)} GPUs[/bold] "
        f"using [bold]{args.ddp} GPU(s) per job[/bold] "
        f"with plans [bold]{plans_file}[/bold]"
    )
    results = schedule_jobs(jobs, gpu_tokens, plans_file, args.trainer, args.disable_tta,
                            args.ckpt_interval, args.disable_train_val, console, args.ddp)
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
