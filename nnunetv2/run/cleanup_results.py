"""Compact nnU-Net training results into a portable, report-oriented layout."""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from nnunetv2.paths import nnUNet_results


METADATA_FILENAMES = ("dataset_fingerprint.json", "dataset.json", "plans.json")


@dataclass(frozen=True)
class Experiment:
    source_name: str
    configuration_name: str
    folds: tuple[str, ...]


@dataclass(frozen=True)
class Dataset:
    source_name: str
    output_name: str
    experiments: tuple[Experiment, ...]


@dataclass
class CleanupSummary:
    datasets: int = 0
    experiments: int = 0
    folds: int = 0
    retained_files: int = 0
    removed_paths: int = 0
    skipped_entries: int = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove unneeded nnU-Net training artifacts and compact the results directory."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Copy cleaned results to this directory. Without it, clean nnUNet_results in place.",
    )
    return parser.parse_args(argv)


def _dataset_output_name(name: str) -> str | None:
    prefix, separator, output_name = name.partition("_")
    if not separator or not prefix.startswith("Dataset") or not prefix.removeprefix("Dataset").isdigit() or not output_name:
        return None
    return output_name


def _configuration_name(name: str) -> str | None:
    parts = name.split("__", 2)
    if len(parts) != 3 or not all(parts):
        return None
    return parts[2]


def _discover_results(root: Path) -> tuple[tuple[Dataset, ...], list[str]]:
    datasets: list[Dataset] = []
    warnings: list[str] = []

    for dataset_path in sorted(root.iterdir(), key=lambda path: path.name):
        if not dataset_path.is_dir():
            warnings.append(f"Skipping unrecognized root entry: {dataset_path}")
            continue

        output_name = _dataset_output_name(dataset_path.name)
        if output_name is None:
            warnings.append(f"Skipping unrecognized dataset directory: {dataset_path}")
            continue

        experiments: list[Experiment] = []
        for experiment_path in sorted(dataset_path.iterdir(), key=lambda path: path.name):
            if not experiment_path.is_dir():
                warnings.append(f"Skipping unrecognized dataset entry: {experiment_path}")
                continue

            configuration_name = _configuration_name(experiment_path.name)
            if configuration_name is None:
                warnings.append(f"Skipping unrecognized experiment directory: {experiment_path}")
                continue

            folds: list[str] = []
            for path in sorted(experiment_path.iterdir(), key=lambda path: path.name):
                if path.is_dir() and path.name.startswith("fold_"):
                    folds.append(path.name)
                elif path.name not in METADATA_FILENAMES:
                    warnings.append(f"Skipping unrecognized experiment entry: {path}")
            if not folds:
                warnings.append(f"Skipping experiment without fold directories: {experiment_path}")
                continue
            experiments.append(Experiment(experiment_path.name, configuration_name, tuple(folds)))

        if experiments:
            datasets.append(Dataset(dataset_path.name, output_name, tuple(experiments)))
        else:
            warnings.append(f"Skipping dataset without recognized experiments: {dataset_path}")

    return tuple(datasets), warnings


def _validate_output_location(source_root: Path, output_root: Path | None) -> None:
    if output_root is None:
        return

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("--output-dir must not be nnUNet_results or a directory inside it.")


def _validate_destinations(datasets: Sequence[Dataset], output_root: Path) -> None:
    claimed_datasets: set[Path] = set()
    claimed_configurations: set[Path] = set()
    claimed_folds: set[Path] = set()

    for dataset in datasets:
        dataset_destination = output_root / dataset.output_name
        if dataset_destination in claimed_datasets:
            raise FileExistsError(f"Multiple source datasets would produce {dataset_destination}.")
        if dataset_destination.exists():
            raise FileExistsError(f"Destination dataset directory already exists: {dataset_destination}")
        claimed_datasets.add(dataset_destination)

        for experiment in dataset.experiments:
            configuration_destination = dataset_destination / experiment.configuration_name
            if configuration_destination in claimed_configurations:
                raise FileExistsError(f"Multiple source experiments would produce {configuration_destination}.")
            claimed_configurations.add(configuration_destination)

            for fold_name in experiment.folds:
                fold_destination = configuration_destination / fold_name
                if fold_destination in claimed_folds:
                    raise FileExistsError(f"Multiple source folds would produce {fold_destination}.")
                claimed_folds.add(fold_destination)


def _validate_fold_summary_destinations(datasets: Sequence[Dataset], source_root: Path) -> None:
    """Avoid replacing an unexpected top-level summary before mutating a fold in place."""
    for dataset in datasets:
        for experiment in dataset.experiments:
            for fold_name in experiment.folds:
                fold = source_root / dataset.source_name / experiment.source_name / fold_name
                validation_summary = fold / "validation" / "summary.json"
                if validation_summary.is_file() and (fold / "summary.json").exists():
                    raise FileExistsError(
                        f"Cannot relocate {validation_summary}: {fold / 'summary.json'} already exists."
                    )


def _retained_files(fold: Path) -> list[Path]:
    files = [fold / name for name in ("checkpoint_last.pth", "progress.png")]
    files.extend(sorted(path for path in fold.glob("training_log_*.txt") if path.is_file()))
    validation_summary = fold / "validation" / "summary.json"
    if validation_summary.is_file():
        files.append(validation_summary)
    return [path for path in files if path.is_file()]


def _remove_path(path: Path) -> int:
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    return 1


def _copy_results(datasets: Sequence[Dataset], source_root: Path, output_root: Path, summary: CleanupSummary) -> None:
    for dataset in datasets:
        for experiment in dataset.experiments:
            for fold_name in experiment.folds:
                source_fold = source_root / dataset.source_name / experiment.source_name / fold_name
                destination_fold = output_root / dataset.output_name / experiment.configuration_name / fold_name
                destination_fold.mkdir(parents=True, exist_ok=False)
                for source_file in _retained_files(source_fold):
                    destination_name = "summary.json" if source_file.name == "summary.json" else source_file.name
                    shutil.copy2(source_file, destination_fold / destination_name)
                    summary.retained_files += 1


def _clean_fold_in_place(fold: Path, summary: CleanupSummary) -> None:
    validation = fold / "validation"
    validation_summary = validation / "summary.json"
    if validation_summary.is_file():
        shutil.move(str(validation_summary), fold / "summary.json")
        summary.retained_files += 1

    retained_names = {"checkpoint_last.pth", "progress.png", "summary.json"}
    for path in fold.iterdir():
        is_training_log = path.name.startswith("training_log_") and path.name.endswith(".txt")
        is_retained = path.is_file() and (path.name in retained_names or is_training_log)
        if is_retained:
            if path.name != "summary.json":
                summary.retained_files += 1
            continue
        summary.removed_paths += _remove_path(path)


def _clean_results_in_place(datasets: Sequence[Dataset], source_root: Path, summary: CleanupSummary) -> None:
    for dataset in datasets:
        source_dataset = source_root / dataset.source_name
        destination_dataset = source_root / dataset.output_name
        source_dataset.rename(destination_dataset)

        for experiment in dataset.experiments:
            source_experiment = destination_dataset / experiment.source_name
            destination_experiment = destination_dataset / experiment.configuration_name
            source_experiment.rename(destination_experiment)

            for fold_name in experiment.folds:
                _clean_fold_in_place(destination_experiment / fold_name, summary)
                summary.folds += 1

            for metadata_filename in METADATA_FILENAMES:
                summary.removed_paths += _remove_path(destination_experiment / metadata_filename)

            summary.experiments += 1
        summary.datasets += 1


def cleanup_results(source_root: Path, output_root: Path | None = None) -> CleanupSummary:
    """Clean recognized results under *source_root*, optionally copying them to *output_root*."""
    source_root = Path(source_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"nnUNet_results directory does not exist: {source_root}")

    if output_root is not None:
        output_root = Path(output_root)
    _validate_output_location(source_root, output_root)

    datasets, warnings = _discover_results(source_root)
    destination_root = output_root if output_root is not None else source_root
    _validate_destinations(datasets, destination_root)
    if output_root is None:
        _validate_fold_summary_destinations(datasets, source_root)

    summary = CleanupSummary(skipped_entries=len(warnings))
    if output_root is None:
        _clean_results_in_place(datasets, source_root, summary)
    else:
        _copy_results(datasets, source_root, output_root, summary)
        summary.datasets = len(datasets)
        summary.experiments = sum(len(dataset.experiments) for dataset in datasets)
        summary.folds = sum(len(experiment.folds) for dataset in datasets for experiment in dataset.experiments)

    for warning in warnings:
        print(f"WARNING: {warning}")
    action = "Copied" if output_root is not None else "Cleaned"
    print(
        f"{action} {summary.datasets} datasets, {summary.experiments} experiments, and {summary.folds} folds; "
        f"retained {summary.retained_files} files, removed {summary.removed_paths} paths, "
        f"skipped {summary.skipped_entries} unrecognized entries."
    )
    return summary


def cleanup_results_entry(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cleanup_results(Path(nnUNet_results), args.output_dir)
    return 0


def main() -> None:
    raise SystemExit(cleanup_results_entry())


if __name__ == "__main__":
    main()
