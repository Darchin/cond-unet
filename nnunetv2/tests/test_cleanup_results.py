import tempfile
import unittest
from pathlib import Path

from nnunetv2.run.cleanup_results import cleanup_results, parse_args


class TestCleanupResults(unittest.TestCase):
    def _make_results(self, root: Path) -> tuple[Path, Path]:
        experiment = root / "Dataset220_KiTS2023" / "nnUNetTrainerAdamW__nnUNetCondUNetPlans__3x-s"
        for name in ("dataset_fingerprint.json", "dataset.json", "plans.json"):
            (experiment / name).parent.mkdir(parents=True, exist_ok=True)
            (experiment / name).write_text("{}")

        fold = experiment / "fold_0"
        validation = fold / "validation"
        validation.mkdir(parents=True)
        for name in ("checkpoint_last.pth", "checkpoint_best.pth", "debug.json", "progress.png", "training_log_a.txt", "training_log_b.txt"):
            (fold / name).write_text(name)
        (validation / "summary.json").write_text("{}")
        (validation / "case_001.nii.gz").write_text("prediction")
        return experiment, fold

    def test_output_mode_copies_only_retained_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results"
            output = Path(tmpdir) / "cleaned"
            experiment, fold = self._make_results(root)

            summary = cleanup_results(root, output)

            cleaned_fold = output / "KiTS2023" / "3x-s" / "fold_0"
            self.assertEqual(
                sorted(path.name for path in cleaned_fold.iterdir()),
                ["checkpoint_last.pth", "progress.png", "summary.json", "training_log_a.txt", "training_log_b.txt"],
            )
            self.assertTrue((experiment / "dataset.json").is_file())
            self.assertTrue((fold / "checkpoint_best.pth").is_file())
            self.assertTrue((fold / "validation" / "case_001.nii.gz").is_file())
            self.assertEqual(summary.retained_files, 5)

    def test_in_place_mode_renames_and_removes_unneeded_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results"
            _, fold = self._make_results(root)
            (root / "notes.txt").write_text("leave me alone")

            summary = cleanup_results(root)

            cleaned_fold = root / "KiTS2023" / "3x-s" / "fold_0"
            self.assertFalse((root / "Dataset220_KiTS2023").exists())
            self.assertTrue((cleaned_fold / "checkpoint_last.pth").is_file())
            self.assertTrue((cleaned_fold / "progress.png").is_file())
            self.assertTrue((cleaned_fold / "training_log_a.txt").is_file())
            self.assertTrue((cleaned_fold / "training_log_b.txt").is_file())
            self.assertTrue((cleaned_fold / "summary.json").is_file())
            self.assertFalse((cleaned_fold / "checkpoint_best.pth").exists())
            self.assertFalse((cleaned_fold / "debug.json").exists())
            self.assertFalse((cleaned_fold / "validation").exists())
            self.assertFalse((root / "KiTS2023" / "3x-s" / "dataset.json").exists())
            self.assertTrue((root / "notes.txt").is_file())
            self.assertEqual(summary.removed_paths, 6)

    def test_destination_collision_aborts_before_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results"
            output = Path(tmpdir) / "cleaned"
            _, fold = self._make_results(root)
            (output / "KiTS2023").mkdir(parents=True)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                cleanup_results(root, output)

            self.assertTrue((fold / "checkpoint_best.pth").is_file())

    def test_partially_completed_fold_is_cleaned_without_optional_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results"
            output = Path(tmpdir) / "cleaned"
            _, fold = self._make_results(root)
            (fold / "checkpoint_last.pth").unlink()
            (fold / "progress.png").unlink()
            (fold / "validation" / "summary.json").unlink()

            cleanup_results(root, output)

            cleaned_fold = output / "KiTS2023" / "3x-s" / "fold_0"
            self.assertEqual(
                sorted(path.name for path in cleaned_fold.iterdir()),
                ["training_log_a.txt", "training_log_b.txt"],
            )

    def test_rejects_output_directory_inside_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results"
            self._make_results(root)

            with self.assertRaisesRegex(ValueError, "inside"):
                cleanup_results(root, root / "cleaned")

    def test_parse_args_accepts_only_output_directory_option(self):
        args = parse_args(["-o", "/tmp/cleaned-results"])
        self.assertEqual(args.output_dir, Path("/tmp/cleaned-results"))


if __name__ == "__main__":
    unittest.main()
