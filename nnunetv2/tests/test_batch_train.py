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
    parse_args,
    requested_configurations,
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

    def test_parse_args_rejects_invalid_job_pair(self):
        with self.assertRaises(SystemExit):
            parse_args(["-d", "1", "-c", "2x", "-f", "0", "-i", "2x"])

    def test_build_jobs_uses_config_then_fold_order(self):
        jobs = build_jobs(["2x", "3x"], [0, 1])

        self.assertEqual([(i.index, i.configuration, i.fold) for i in jobs],
                         [(0, "2x", 0), (1, "2x", 1), (2, "3x", 0), (3, "3x", 1)])
        self.assertEqual([i.total for i in jobs], [4, 4, 4, 4])

    def test_apply_job_overrides_appends_included_jobs_without_duplicates(self):
        jobs = apply_job_overrides(
            build_jobs(["3x-s", "4x-m"], [0]),
            include=[("3x-s", 0), ("4x-m", 1), ("2x-xs", 3)],
        )

        self.assertEqual(
            [(i.index, i.total, i.configuration, i.fold) for i in jobs],
            [
                (0, 4, "3x-s", 0),
                (1, 4, "4x-m", 0),
                (2, 4, "4x-m", 1),
                (3, 4, "2x-xs", 3),
            ],
        )

    def test_apply_job_overrides_removes_excluded_jobs_after_include(self):
        jobs = apply_job_overrides(
            build_jobs(["3x-s", "4x-m"], [0, 1]),
            include=[("5x-l", 2), ("3x-s", 1)],
            exclude=[("3x-s", 1), ("5x-l", 2), ("unused", 0)],
        )

        self.assertEqual(
            [(i.index, i.total, i.configuration, i.fold) for i in jobs],
            [
                (0, 3, "3x-s", 0),
                (1, 3, "4x-m", 0),
                (2, 3, "4x-m", 1),
            ],
        )

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
