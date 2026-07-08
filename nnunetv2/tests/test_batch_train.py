import os
import tempfile
import unittest
from unittest.mock import patch

from batchgenerators.utilities.file_and_folder_operations import save_json
from rich.console import Console

from nnunetv2.run.batch_train import (
    apply_job_overrides,
    build_jobs,
    format_duration,
    load_job_pairs_from_json,
    parse_args,
    requested_configurations,
    resolve_cli_job_pairs,
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
                    {"configs": "4x-m", "folds": 4},
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

    def test_load_job_pairs_from_json_rejects_overlap_across_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = os.path.join(tmpdir, "jobs.json")
            save_json(
                [
                    {"configs": ["3x-s", "3x-m"], "folds": [0]},
                    {"configs": "3x-s", "folds": 0},
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
            [{"configs": [], "folds": 0}],
            [{"configs": "3x-s", "folds": []}],
            [{"configs": "", "folds": 0}],
            [{"configs": [1], "folds": 0}],
            [{"configs": "3x-s", "folds": [True]}],
            [{"configs": "3x-s"}],
            [{"configs": "3x-s", "folds": 0, "priority": 1}],
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
