import tempfile
import unittest
from pathlib import Path

from nnunetv2.run.run_training import maybe_load_checkpoint, run_training
from nnunetv2.training.logging.nnunet_logger import LocalLogger, MetaLogger
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class _CheckpointLoader:
    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        self.loaded = []

    def load_checkpoint(self, filename: str) -> None:
        self.loaded.append(filename)


class TestTrainingBehavior(unittest.TestCase):
    def test_resume_loads_checkpoint_last_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint_last.pth"
            checkpoint.write_bytes(b"checkpoint")
            trainer = _CheckpointLoader(tmpdir)

            maybe_load_checkpoint(trainer, continue_training=True, validation_only=False)

            self.assertEqual(trainer.loaded, [str(checkpoint)])

    def test_resume_does_not_fall_back_to_legacy_checkpoint_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "checkpoint_final.pth").write_bytes(b"legacy")
            trainer = _CheckpointLoader(tmpdir)

            maybe_load_checkpoint(trainer, continue_training=True, validation_only=False)

            self.assertEqual(trainer.loaded, [])

    def test_programmatic_api_rejects_non_positive_checkpoint_interval(self):
        with self.assertRaisesRegex(ValueError, "checkpoint_interval"):
            run_training("1", "2d", "0", checkpoint_interval=0)

    def test_disabled_training_validation_skips_validation_hooks_and_steps(self):
        events = []

        class FakeTrainer:
            current_epoch = 0
            num_epochs = 1
            num_iterations_per_epoch = 2
            num_val_iterations_per_epoch = 3
            disable_train_val = True

            def on_train_start(self):
                events.append("train_start")
                self.dataloader_train = iter((1, 2))

            def on_epoch_start(self):
                events.append("epoch_start")

            def on_train_epoch_start(self):
                events.append("train_epoch_start")

            def train_step(self, batch):
                events.append(f"train_step_{batch}")
                return {"loss": batch}

            def on_train_epoch_end(self, outputs):
                events.append(f"train_epoch_end_{len(outputs)}")

            def on_validation_epoch_start(self):
                raise AssertionError("training validation must be skipped")

            def validation_step(self, batch):
                raise AssertionError("training validation must be skipped")

            def on_validation_epoch_end(self, outputs):
                raise AssertionError("training validation must be skipped")

            def on_epoch_end(self):
                events.append("epoch_end")

            def on_train_end(self):
                events.append("train_end")

        nnUNetTrainer.run_training(FakeTrainer())

        self.assertEqual(
            events,
            [
                "train_start",
                "epoch_start",
                "train_epoch_start",
                "train_step_1",
                "train_step_2",
                "train_epoch_end_2",
                "epoch_end",
                "train_end",
            ],
        )

    def test_interval_checkpoint_uses_checkpoint_last_name(self):
        class FakeLogger:
            def get_value(self, key, step):
                return {
                    "train_losses": 1.0,
                    "epoch_start_timestamps": 10.0,
                    "epoch_end_timestamps": 12.0,
                }[key]

            def log(self, key, value, epoch):
                pass

            def plot_progress_png(self, output_folder, include_validation):
                pass

        class FakeTrainer:
            current_epoch = 49
            num_epochs = 100
            checkpoint_interval = 50
            disable_train_val = True
            output_folder = "/results/fold_0"
            local_rank = 0
            logger = FakeLogger()

            def print_to_log_file(self, *args, **kwargs):
                pass

            def save_checkpoint(self, filename):
                self.saved_checkpoint = filename

        trainer = FakeTrainer()

        nnUNetTrainer.on_epoch_end(trainer)

        self.assertEqual(trainer.saved_checkpoint, "/results/fold_0/checkpoint_last.pth")
        self.assertEqual(trainer.current_epoch, 50)

    def test_training_only_progress_plot_does_not_require_validation_metrics(self):
        logger = LocalLogger()
        logger.log("train_losses", 1.0, 0)
        logger.log("lrs", 0.01, 0)
        logger.log("epoch_start_timestamps", 10.0, 0)
        logger.log("epoch_end_timestamps", 12.0, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.plot_progress_png(tmpdir, include_validation=False)

            self.assertTrue((Path(tmpdir) / "progress.png").is_file())

    def test_validation_metrics_can_resume_after_disabled_epochs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetaLogger(tmpdir, resume=False)

            logger.log("mean_fg_dice", 0.5, 2)

            self.assertEqual(logger.get_value("mean_fg_dice", step=None), [None, None, 0.5])
            self.assertEqual(logger.get_value("ema_fg_dice", step=None), [None, None, 0.5])


if __name__ == "__main__":
    unittest.main()
