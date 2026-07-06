from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from os.path import abspath, dirname, isfile, join
from typing import Callable, Sequence

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

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
class JobProgress:
    stage: str = "training"
    training_completed: int = 0
    training_total: int | None = None
    validation_completed: int = 0
    validation_total: int | None = None
    training_elapsed_seconds: float | None = None
    validation_elapsed_seconds: float | None = None


@dataclass
class RunningJob:
    job: TrainingJob
    gpu: str
    process: subprocess.Popen
    start_time: float
    progress_file: str
    progress_file_offset: int = 0
    progress: JobProgress = field(default_factory=JobProgress)


@dataclass(frozen=True)
class FinishedJob:
    job: TrainingJob
    gpu: str
    returncode: int
    elapsed_seconds: float
    progress: JobProgress = field(default_factory=JobProgress)


class BatchTrainingAborted(RuntimeError):
    pass


JobPair = tuple[str, int]


class ProgressEventWriter:
    def __init__(self, progress_file: str, monotonic: Callable[[], float] = time.monotonic):
        self.progress_file = progress_file
        self.monotonic = monotonic
        self.stage_start_times: dict[str, float] = {}

    def __call__(self, event: str, completed: int | None = None, total: int | None = None) -> None:
        now = self.monotonic()
        stage = event.split("_", maxsplit=1)[0]
        payload: dict[str, str | int | float] = {"event": event}
        if completed is not None:
            payload["completed"] = completed
        if total is not None:
            payload["total"] = total
        if event.endswith("_started"):
            self.stage_start_times[stage] = now
        if event.endswith("_finished"):
            payload["elapsed_seconds"] = now - self.stage_start_times.get(stage, now)

        with open(self.progress_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")


def format_duration(seconds: float) -> str:
    minutes = max(0, int(seconds // 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def job_header(status_markup: str, job: TrainingJob, gpu: str) -> str:
    return (
        f"{status_markup} [bold]job {job.index}/{job.total - 1}[/bold] on GPU [bold]{escape(gpu)}[/bold], "
        f"config=[bold]{escape(job.configuration)}[/bold], fold=[bold]{job.fold}[/bold]"
    )


def render_progress_row(progress: JobProgress):
    if progress.stage == "validation":
        label = Text.from_markup("[bold blue]VALIDATION[/bold blue]")
        completed = progress.validation_completed
        total = progress.validation_total
        style = "blue"
    else:
        label = Text.from_markup("[bold yellow]TRAINING[/bold yellow]")
        completed = progress.training_completed
        total = progress.training_total
        style = "yellow"

    visible_total = max(total or 1, 1)
    visible_completed = min(max(completed, 0), visible_total)
    count = f"{completed}/{total}" if total is not None else f"{completed}/?"

    row = Table.grid(expand=True)
    row.add_column(width=11)
    row.add_column(ratio=1)
    row.add_column(width=10, justify="right")
    row.add_row(
        label,
        ProgressBar(
            total=visible_total,
            completed=visible_completed,
            width=None,
            complete_style=style,
            finished_style=style,
            pulse_style=style,
        ),
        Text(count),
    )
    return row


def render_running_panel(running_job: RunningJob) -> Panel:
    return render_start_panel(running_job.job, running_job.gpu, running_job.progress)


def render_start_panel(job: TrainingJob, gpu: str, progress: JobProgress | None = None) -> Panel:
    return Panel(
        Group(
            Text.from_markup(job_header("[bold cyan]START[/bold cyan]", job, gpu)),
            render_progress_row(progress or JobProgress()),
        ),
        border_style="cyan",
    )


def render_finished_panel(result: FinishedJob) -> Panel:
    if result.returncode == 0:
        status = "[bold green]DONE[/bold green]"
        border_style = "green"
    else:
        status = f"[bold red]FAILED[/bold red] returncode=[bold]{result.returncode}[/bold]"
        border_style = "red"

    progress = result.progress
    if progress.training_elapsed_seconds is None:
        training = "[bold yellow]TRAINING[/bold yellow] not completed"
    else:
        training = (
            f"[bold yellow]TRAINING[/bold yellow] finished in "
            f"[bold]{format_duration(progress.training_elapsed_seconds)}[/bold]"
        )
    if progress.validation_elapsed_seconds is None:
        validation = "[bold blue]VALIDATION[/bold blue] not completed"
    else:
        validation = (
            f"[bold blue]VALIDATION[/bold blue] finished in "
            f"[bold]{format_duration(progress.validation_elapsed_seconds)}[/bold]"
        )

    return Panel(
        Group(
            Text.from_markup(job_header(status, result.job, result.gpu)),
            Text.from_markup(f"{training}    {validation}"),
        ),
        border_style=border_style,
    )


def render_abort_panel(running: Sequence[RunningJob]) -> Panel:
    jobs = ", ".join(str(i.job.index) for i in running) if running else "none"
    return Panel(
        Text.from_markup(
            f"[bold red]ABORT[/bold red] Keyboard interrupt received; gracefully terminating running jobs "
            f"and stopping. running_jobs=[bold]{jobs}[/bold]"
        ),
        border_style="red",
    )


def render_running_jobs(running: Sequence[RunningJob]) -> Group:
    return Group(*(render_running_panel(i) for i in running))


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
                        disable_tta: bool,
                        progress_file: str | None = None) -> None:
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
    if progress_file is not None:
        trainer.progress_callback = ProgressEventWriter(progress_file)

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
                        disable_tta: bool,
                        progress_file: str | None = None) -> list[str]:
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
    if progress_file is not None:
        command += ["--progress-file", progress_file]
    return command


def launch_job(job: TrainingJob,
               gpu: str,
               plans_file: str,
               trainer_name: str,
               disable_tta: bool,
               progress_file: str | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    return subprocess.Popen(
        make_worker_command(plans_file, job.configuration, job.fold, trainer_name, disable_tta, progress_file),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )


def log_start(console: Console, job: TrainingJob, gpu: str) -> None:
    console.print(render_start_panel(job, gpu))


def log_finish(console: Console, result: FinishedJob) -> None:
    console.print(render_finished_panel(result))


def create_progress_file() -> str:
    progress_file = tempfile.NamedTemporaryFile(prefix="nnunet_batch_train_", suffix=".jsonl", delete=False)
    try:
        return progress_file.name
    finally:
        progress_file.close()


def cleanup_progress_file(progress_file: str) -> None:
    if not progress_file:
        return
    try:
        os.unlink(progress_file)
    except FileNotFoundError:
        pass


def read_progress_events(running_job: RunningJob) -> list[dict]:
    if not running_job.progress_file or not isfile(running_job.progress_file):
        return []

    events = []
    with open(running_job.progress_file, "r", encoding="utf-8") as f:
        f.seek(running_job.progress_file_offset)
        while True:
            line = f.readline()
            if not line:
                break
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        running_job.progress_file_offset = f.tell()
    return events


def apply_progress_event(progress: JobProgress, event: dict) -> None:
    event_name = event.get("event")
    if not isinstance(event_name, str):
        return

    if event_name.startswith("training_"):
        progress.stage = "training"
        if isinstance(event.get("completed"), int):
            progress.training_completed = event["completed"]
        if isinstance(event.get("total"), int):
            progress.training_total = event["total"]
        if event_name == "training_finished" and isinstance(event.get("elapsed_seconds"), (int, float)):
            progress.training_elapsed_seconds = float(event["elapsed_seconds"])
    elif event_name.startswith("validation_"):
        progress.stage = "validation"
        if isinstance(event.get("completed"), int):
            progress.validation_completed = event["completed"]
        if isinstance(event.get("total"), int):
            progress.validation_total = event["total"]
        if event_name == "validation_finished" and isinstance(event.get("elapsed_seconds"), (int, float)):
            progress.validation_elapsed_seconds = float(event["elapsed_seconds"])


def refresh_progress(running: Sequence[RunningJob]) -> None:
    for running_job in running:
        for event in read_progress_events(running_job):
            apply_progress_event(running_job.progress, event)


def signal_process_group(process: subprocess.Popen, sig: signal.Signals) -> None:
    pid = getattr(process, "pid", None)
    if pid is not None:
        try:
            os.killpg(pid, sig)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass

    if sig == signal.SIGTERM and hasattr(process, "terminate"):
        process.terminate()
    elif sig == signal.SIGKILL and hasattr(process, "kill"):
        process.kill()


def terminate_running_jobs(running: Sequence[RunningJob],
                           monotonic: Callable[[], float] = time.monotonic,
                           sleep: Callable[[float], None] = time.sleep,
                           grace_seconds: float = 15.0) -> None:
    for running_job in running:
        signal_process_group(running_job.process, signal.SIGTERM)

    deadline = monotonic() + grace_seconds
    while monotonic() < deadline:
        if all(running_job.process.poll() is not None for running_job in running):
            return
        sleep(0.2)

    for running_job in running:
        if running_job.process.poll() is None:
            signal_process_group(running_job.process, signal.SIGKILL)


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
                  launcher: Callable[[TrainingJob, str, str, str, bool, str | None], subprocess.Popen] = launch_job,
                  monotonic: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep,
                  poll_interval: float = 1.0) -> list[FinishedJob]:
    pending = list(jobs)
    free_gpus = list(gpu_tokens)
    running: list[RunningJob] = []
    results: list[FinishedJob] = []
    progress_files: list[str] = []

    try:
        with Live(render_running_jobs(running), console=console, refresh_per_second=4, transient=False) as live:
            while pending or running:
                while pending and free_gpus:
                    gpu = free_gpus.pop(0)
                    job = pending.pop(0)
                    progress_file = create_progress_file()
                    progress_files.append(progress_file)
                    process = launcher(job, gpu, plans_file, trainer_name, disable_tta, progress_file)
                    running.append(RunningJob(job, gpu, process, monotonic(), progress_file))

                refresh_progress(running)

                finished = []
                for running_job in running:
                    returncode = running_job.process.poll()
                    if returncode is not None:
                        for event in read_progress_events(running_job):
                            apply_progress_event(running_job.progress, event)
                        result = FinishedJob(
                            running_job.job,
                            running_job.gpu,
                            returncode,
                            monotonic() - running_job.start_time,
                            running_job.progress,
                        )
                        results.append(result)
                        finished.append(running_job)
                        free_gpus.append(running_job.gpu)
                        live.console.print(render_finished_panel(result))

                if finished:
                    running = [i for i in running if i not in finished]
                    for running_job in finished:
                        cleanup_progress_file(running_job.progress_file)

                live.update(render_running_jobs(running))

                if not finished and running:
                    sleep(poll_interval)
    except KeyboardInterrupt as exc:
        console.print(render_abort_panel(running))
        terminate_running_jobs(running, monotonic=monotonic, sleep=sleep)
        raise BatchTrainingAborted from exc
    finally:
        for progress_file in progress_files:
            cleanup_progress_file(progress_file)

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
    parser.add_argument("--progress-file", help=argparse.SUPPRESS)
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
        run_training_worker(
            args.plans_file,
            args.configuration,
            args.fold,
            args.trainer,
            args.disable_tta,
            args.progress_file,
        )
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
    try:
        results = schedule_jobs(jobs, gpu_tokens, plans_file, args.trainer, args.disable_tta, console)
    except BatchTrainingAborted:
        return 130
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
