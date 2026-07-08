import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

from batchgenerators.utilities.file_and_folder_operations import save_json
from rich.console import Console

from nnunetv2.run.batch_train import (
    FinishedJob,
    apply_job_overrides,
    build_jobs,
    format_duration,
    format_finish_message,
    format_start_message,
    format_verbose_duration,
    load_job_pairs_from_json,
    parse_args,
    requested_configurations,
    resolve_cli_job_pairs,
    resolve_job_json_file,
    resolve_plans_file,
    schedule_jobs,
    validate_requested_configurations,
    visible_gpu_tokens,
)


class FakeProcess:
    def __init__(self, returncode=0, running_polls=0):
        self.returncode = returncode
        self.running_polls = running_polls

    def poll(self):
        if self.running_polls > 0:
            self.running_polls -= 1
            return None
        return self.returncode


class TestBatchTrain(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = parse_args(["-d", "1", "-c", "2x", "-f", "0"])

        self.assertEqual(args.plan, "nnUNetCondUNetPlans")
        self.assertEqual(args.trainer, "nnUNetTrainerAdamW")
        self.assertFalse(args.disable_tta)

    def test_parse_args_disable_tta(self):
        args = parse_args(["-d", "1", "-t", "nnUNetTrainer", "-c", "2x", "-f", "0", "--disable-tta"])

        self.assertTrue(args.disable_tta)

    def test_parse_args_include_and_exclude_job_pairs(self):
        args = parse_args([
            "-d", "1",
            "-c", "3x-s", "4x-m",
            "-f", "0", "1",
            "-i", "3x-s,0", "4x-m,1",
            "-e", "3x-s,1",
        ])

        self.assertEqual(args.include, [("3x-s", 0), ("4x-m", 1)])
        self.assertEqual(args.exclude, [("3x-s", 1)])

    def test_parse_args_accepts_include_without_configs_or_folds(self):
        args = parse_args(["-d", "1", "-i", "3x-s,0"])

        self.assertIsNone(args.configs)
        self.assertIsNone(args.folds)
        self.assertEqual(args.include, [("3x-s", 0)])

    def test_parse_args_accepts_json_without_configs_or_folds(self):
        args = parse_args(["-d", "1", "-j", "jobs.json"])

        self.assertEqual(args.json, "jobs.json")
        self.assertIsNone(args.configs)
        self.assertIsNone(args.folds)

    def test_parse_args_rejects_json_with_cli_job_arguments(self):
        with self.assertRaises(SystemExit):
            parse_args(["-d", "1", "-j", "jobs.json", "-c", "2x", "-f", "0"])

    def test_parse_args_rejects_invalid_job_pair(self):
        with self.assertRaises(SystemExit):
            parse_args(["-d", "1", "-c", "2x", "-f", "0", "-i", "2x"])

    def test_build_jobs_uses_config_then_fold_order(self):
        jobs = build_jobs(["2x", "3x"], [0, 1])

        self.assertEqual([(i.index, i.configuration, i.fold) for i in jobs],
                         [(0, "2x", 0), (1, "2x", 1), (2, "3x", 0), (3, "3x", 1)])
        self.assertEqual([i.total for i in jobs], [4, 4, 4, 4])

    def test_apply_job_overrides_appends_included_jobs(self):
        jobs = apply_job_overrides(build_jobs(["3x-s", "4x-m"], [0]), include=[("4x-m", 1), ("2x-xs", 3)])

        self.assertEqual(
            [(i.index, i.total, i.configuration, i.fold) for i in jobs],
            [
                (0, 4, "3x-s", 0),
                (1, 4, "4x-m", 0),
                (2, 4, "4x-m", 1),
                (3, 4, "2x-xs", 3),
            ],
        )

    def test_apply_job_overrides_removes_excluded_jobs_before_include(self):
        jobs = apply_job_overrides(
            build_jobs(["3x-s", "4x-m"], [0, 1]),
            include=[("5x-l", 2)],
            exclude=[("3x-s", 1)],
        )

        self.assertEqual(
            [(i.index, i.total, i.configuration, i.fold) for i in jobs],
            [
                (0, 4, "3x-s", 0),
                (1, 4, "4x-m", 0),
                (2, 4, "4x-m", 1),
                (3, 4, "5x-l", 2),
            ],
        )

    def test_resolve_cli_job_pairs_accepts_include_only(self):
        self.assertEqual(resolve_cli_job_pairs(None, None, include=[("3x-s", 0)]), [("3x-s", 0)])

    def test_resolve_cli_job_pairs_rejects_empty_jobs(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            resolve_cli_job_pairs(None, None)

    def test_resolve_cli_job_pairs_rejects_configs_without_folds(self):
        with self.assertRaisesRegex(ValueError, "together"):
            resolve_cli_job_pairs(["3x-s"], None)

    def test_resolve_cli_job_pairs_rejects_include_already_in_original_jobs(self):
        with self.assertRaisesRegex(ValueError, "already exist"):
            resolve_cli_job_pairs(["3x-s"], [0], include=[("3x-s", 0)])

    def test_resolve_cli_job_pairs_rejects_exclude_missing_from_original_jobs(self):
        with self.assertRaisesRegex(ValueError, "do not exist"):
            resolve_cli_job_pairs(["3x-s"], [0], exclude=[("4x-m", 0)])

    def test_resolve_cli_job_pairs_rejects_include_exclude_overlap(self):
        with self.assertRaisesRegex(ValueError, "both include and exclude"):
            resolve_cli_job_pairs(["3x-s"], [0], include=[("4x-m", 0)], exclude=[("4x-m", 0)])

    def test_load_job_pairs_from_json_expands_ordered_job_specs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = os.path.join(tmpdir, "jobs.json")
            save_json(
                [
                    {"configs": ["3x-s", "3x-m"], "folds": [0, 1, 2]},
                    {"configs": ["4x-m"], "folds": [4]},
                ],
                json_file,
            )

            self.assertEqual(
                load_job_pairs_from_json(json_file),
                [
                    ("3x-s", 0),
                    ("3x-s", 1),
                    ("3x-s", 2),
                    ("3x-m", 0),
                    ("3x-m", 1),
                    ("3x-m", 2),
                    ("4x-m", 4),
                ],
            )

    def test_resolve_job_json_file_resolves_bare_names_under_jobs_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = os.path.join(tmpdir, "jobs")
            fallback_dir = os.path.join(tmpdir, "fallback")
            os.makedirs(jobs_dir)
            os.makedirs(fallback_dir)
            jobs_file = os.path.join(jobs_dir, "phase-one.json")
            fallback_file = os.path.join(fallback_dir, "phase-one.json")
            save_json([], jobs_file)
            save_json([], fallback_file)

            with patch("nnunetv2.run.batch_train.JOBS_DIR", jobs_dir), \
                    patch.object(os, "getcwd", return_value=fallback_dir):
                self.assertEqual(resolve_job_json_file("phase-one"), jobs_file)

    def test_resolve_job_json_file_treats_json_names_as_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = os.path.join(tmpdir, "jobs")
            fallback_dir = os.path.join(tmpdir, "fallback")
            os.makedirs(jobs_dir)
            os.makedirs(fallback_dir)
            jobs_file = os.path.join(jobs_dir, "phase-one.json")
            fallback_file = os.path.join(fallback_dir, "phase-one.json")
            save_json([], jobs_file)
            save_json([], fallback_file)

            with patch("nnunetv2.run.batch_train.JOBS_DIR", jobs_dir), \
                    patch.object(os, "getcwd", return_value=fallback_dir):
                self.assertEqual(resolve_job_json_file("phase-one.json"), fallback_file)

    def test_resolve_job_json_file_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("nnunetv2.run.batch_train.JOBS_DIR", os.path.join(tmpdir, "jobs")):
                with self.assertRaises(FileNotFoundError):
                    resolve_job_json_file("missing.json")

    def test_load_job_pairs_from_json_rejects_overlap_across_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = os.path.join(tmpdir, "jobs.json")
            save_json(
                [
                    {"configs": ["3x-s", "3x-m"], "folds": [0]},
                    {"configs": ["3x-s"], "folds": [0]},
                ],
                json_file,
            )

            with self.assertRaisesRegex(ValueError, "overlaps"):
                load_job_pairs_from_json(json_file)

    def test_load_job_pairs_from_json_rejects_duplicate_pairs_inside_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = os.path.join(tmpdir, "jobs.json")
            save_json([{"configs": ["3x-s", "3x-s"], "folds": [0]}], json_file)

            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_job_pairs_from_json(json_file)

    def test_load_job_pairs_from_json_rejects_invalid_shapes(self):
        invalid_specs = [
            {"configs": "3x-s", "folds": 0},
            [{"configs": [], "folds": [0]}],
            [{"configs": ["3x-s"], "folds": []}],
            [{"configs": [""], "folds": [0]}],
            [{"configs": [1], "folds": [0]}],
            [{"configs": ["3x-s"], "folds": [True]}],
            [{"configs": "3x-s"}],
            [{"configs": ["3x-s"], "folds": [0], "priority": 1}],
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for index, spec in enumerate(invalid_specs):
                json_file = os.path.join(tmpdir, f"jobs_{index}.json")
                save_json(spec, json_file)

                with self.assertRaises(ValueError):
                    load_job_pairs_from_json(json_file)

    def test_requested_configurations_includes_override_configs_once(self):
        self.assertEqual(
            requested_configurations(
                ["3x-s", "4x-m"],
                include=[("4x-m", 1), ("5x-l", 0)],
                exclude=[("3x-s", 2), ("6x-xl", 0)],
            ),
            ["3x-s", "4x-m", "5x-l", "6x-xl"],
        )

    def test_format_duration(self):
        self.assertEqual(format_duration(0), "0h 0m")
        self.assertEqual(format_duration(59), "0h 0m")
        self.assertEqual(format_duration(61), "0h 1m")
        self.assertEqual(format_duration(7260), "2h 1m")

    def test_format_verbose_duration_pluralizes_hours_and_minutes(self):
        self.assertEqual(format_verbose_duration(0), "[bold]0[/bold] hours and [bold]0[/bold] minutes")
        self.assertEqual(format_verbose_duration(60), "[bold]0[/bold] hours and [bold]1[/bold] minute")
        self.assertEqual(format_verbose_duration(3600), "[bold]1[/bold] hour and [bold]0[/bold] minutes")
        self.assertEqual(format_verbose_duration(7260), "[bold]2[/bold] hours and [bold]1[/bold] minute")

    def test_format_start_message_uses_single_line_classic_style(self):
        job = build_jobs(["3x-m"], [3])[0]

        self.assertEqual(
            format_start_message(job, "2", datetime(2026, 7, 8, 16, 33)),
            "[bold cyan]STARTED[/bold cyan] job [bold]1/1[/bold] on GPU [bold]2[/bold] — "
            "fold [bold]3[/bold] of config [bold]3x-m[/bold] — "
            "started on [bold]2026-07-08 at 16:33[/bold].",
        )

    def test_format_finish_message_uses_finished_times_and_duration(self):
        job = build_jobs(["3x-m"], [3])[0]
        result = FinishedJob(
            job=job,
            gpu="2",
            returncode=0,
            elapsed_seconds=11640,
            started_at=datetime(2026, 7, 8, 16, 33),
            finished_at=datetime(2026, 7, 8, 19, 47),
        )

        self.assertEqual(
            format_finish_message(result),
            "[bold green]FINISHED[/bold green] job [bold]1/1[/bold] on GPU [bold]2[/bold] — "
            "fold [bold]3[/bold] of config [bold]3x-m[/bold] — "
            "started on [bold]2026-07-08 at 16:33[/bold], "
            "finished on [bold]2026-07-08 at 19:47[/bold], "
            "took [bold]3[/bold] hours and [bold]14[/bold] minutes in total.",
        )

    def test_format_finish_message_uses_failed_status(self):
        job = build_jobs(["3x-m"], [3])[0]
        result = FinishedJob(
            job=job,
            gpu="2",
            returncode=1,
            elapsed_seconds=60,
            started_at=datetime(2026, 7, 8, 16, 33),
            finished_at=datetime(2026, 7, 8, 16, 34),
        )

        self.assertEqual(
            format_finish_message(result),
            "[bold red]FAILED[/bold red] job [bold]1/1[/bold] on GPU [bold]2[/bold] — "
            "fold [bold]3[/bold] of config [bold]3x-m[/bold] — "
            "started on [bold]2026-07-08 at 16:33[/bold], "
            "finished on [bold]2026-07-08 at 16:34[/bold], "
            "took [bold]0[/bold] hours and [bold]1[/bold] minute in total.",
        )

    def test_visible_gpu_tokens_respects_existing_cuda_visible_devices(self):
        self.assertEqual(visible_gpu_tokens("2, 4", 2), ["2", "4"])
        self.assertEqual(visible_gpu_tokens(None, 3), ["0", "1", "2"])

    def test_resolve_plan_path_accepts_direct_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plans_file = os.path.join(tmpdir, "plans.json")
            save_json({"configurations": {}}, plans_file)

            self.assertEqual(resolve_plans_file("1", plans_file), plans_file)

    def test_resolve_plan_identifier_uses_preprocessed_dataset_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(tmpdir, "Dataset001_Test")
            os.makedirs(dataset_dir)
            plans_file = os.path.join(dataset_dir, "nnUNetCondUNetPlans.json")
            save_json({"configurations": {}}, plans_file)

            with patch.dict(os.environ, {"nnUNet_preprocessed": tmpdir}, clear=False):
                self.assertEqual(resolve_plans_file("Dataset001_Test", "nnUNetCondUNetPlans"), plans_file)

    def test_validate_requested_configurations_rejects_missing_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plans_file = os.path.join(tmpdir, "plans.json")
            save_json({"dataset_name": "Dataset001_Test", "plans_name": "plans", "configurations": {"2x": {}}},
                      plans_file)

            with self.assertRaisesRegex(RuntimeError, "3x"):
                validate_requested_configurations(plans_file, ["2x", "3x"])

    def test_scheduler_reuses_gpus_and_continues_after_failure(self):
        jobs = build_jobs(["2x", "3x"], [0, 1])
        launches = []
        returncodes = [0, 1, 0, 0]

        def launcher(job, gpu, plans_file, trainer_name, disable_tta):
            launches.append((job.index, gpu, plans_file, trainer_name, disable_tta))
            return FakeProcess(returncodes[job.index])

        now = {"value": 0.0}

        def monotonic():
            now["value"] += 60.0
            return now["value"]

        console = Console(file=open(os.devnull, "w"))
        try:
            results = schedule_jobs(
                jobs,
                ["0", "1"],
                "plans.json",
                "nnUNetTrainer",
                True,
                console,
                launcher=launcher,
                monotonic=monotonic,
                sleep=lambda _: None,
            )
        finally:
            console.file.close()

        self.assertEqual([i[0] for i in launches], [0, 1, 2, 3])
        self.assertEqual([i[1] for i in launches], ["0", "1", "0", "1"])
        self.assertTrue(all(i[4] for i in launches))
        self.assertEqual([i.returncode for i in results], [0, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
