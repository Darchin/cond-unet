import unittest

from nnunetv2.experiment_planning.dataset_fingerprint.cond_unet_fingerprint_extractor import (
    CondUNetFingerprintExtractor,
)
from nnunetv2.experiment_planning.experiment_planners.cond_unet_planner import CondUNetPlanner
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerAdamW import nnUNetTrainerAdamW
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


class TestCondUNetPlannerHelpers(unittest.TestCase):
    def test_presets_use_expected_stage_depths(self):
        self.assertEqual(CondUNetPlanner.presets["2x"]["post_stem_downsampling_stages"], 4)
        self.assertEqual(CondUNetPlanner.presets["3x"]["post_stem_downsampling_stages"], 3)
        self.assertEqual(CondUNetPlanner.presets["4x"]["post_stem_downsampling_stages"], 3)

    def test_spacing_percentiles_are_json_friendly(self):
        percentiles = CondUNetFingerprintExtractor.compute_spacing_percentiles(
            [[1.0, 5.0, 1.0], [1.0, 3.0, 1.0]]
        )

        self.assertIn("50", percentiles)
        self.assertEqual(percentiles["50"], [1.0, 4.0, 1.0])

    def test_target_spacing_selects_percentile_before_crossing(self):
        spacing_percentiles = {
            "50": [1.0, 5.0, 1.0],
            "45": [1.0, 4.0, 1.0],
            "40": [1.0, 2.5, 1.0],
            "35": [1.0, 2.0, 1.0],
            "30": [1.0, 2.0, 1.0],
            "25": [1.0, 2.0, 1.0],
        }

        target_spacing = CondUNetPlanner.determine_target_spacing_for_factor(spacing_percentiles, 3)

        self.assertEqual(target_spacing.tolist(), [1.0, 4.0, 1.0])

    def test_stem_stride_keeps_reference_axes_at_factor(self):
        stem_stride = CondUNetPlanner.determine_stem_stride(
            median_spacing=[1.0, 5.0, 1.0],
            target_spacing=[1.0, 4.0, 1.0],
            stem_factor=3,
        )

        self.assertEqual(stem_stride.tolist(), [3, 1, 3])

    def test_patch_geometry_uses_physical_aspect_ratio_and_total_stride(self):
        aspect_ratio, patch_size_unit = CondUNetPlanner.compute_patch_geometry(
            target_spacing=[4.0, 1.0, 1.0],
            median_shape=[10, 20, 40],
            stem_stride=[1, 3, 3],
            post_stem_downsampling_stages=4,
        )

        self.assertEqual(aspect_ratio, [2, 1, 2])
        self.assertEqual(patch_size_unit, [32, 48, 96])

    def test_architecture_uses_relu(self):
        planner = CondUNetPlanner.__new__(CondUNetPlanner)

        architecture = planner._architecture(
            dim=3,
            n_stages=5,
            post_stem_downsampling_stages=4,
            stem_stride=[2, 2, 2],
            stem_kernel_size=[3, 3, 3],
        )

        self.assertEqual(architecture["arch_kwargs"]["nonlin"], "torch.nn.ReLU")
        self.assertEqual(architecture["arch_kwargs"]["nonlin_kwargs"], {"inplace": True})
        self.assertEqual(architecture["arch_kwargs"]["n_stages"], 5)
        self.assertEqual(len(architecture["arch_kwargs"]["strides"]), 5)


class TestPlansManagerCondUNetConfig(unittest.TestCase):
    def test_preset_inherits_shared_preprocessing_keys_from_base(self):
        plans = {
            "configurations": {
                "base": {
                    "normalization_schemes": ["ZScoreNormalization"],
                    "use_mask_for_norm": [False],
                    "resampling_fn_data": "resample_data_or_seg_to_shape",
                    "resampling_fn_seg": "resample_data_or_seg_to_shape",
                    "resampling_fn_data_kwargs": {"is_seg": False},
                    "resampling_fn_seg_kwargs": {"is_seg": True},
                    "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
                    "resampling_fn_probabilities_kwargs": {"is_seg": False},
                },
                "2x": {
                    "inherits_from": "base",
                    "data_identifier": "two_x",
                    "preprocessor_name": "DefaultPreprocessor",
                    "batch_size": 2,
                    "patch_size": [16, 32, 48],
                    "patch_size_unit": [16, 32, 48],
                    "patch_size_multiplier": None,
                    "median_image_size_in_voxels": [128, 128, 128],
                    "spacing": [1.0, 1.0, 1.0],
                    "batch_dice": True,
                    "architecture": {
                        "network_class_name": "ParentNet",
                        "arch_kwargs": {"features_per_stage": None},
                        "_kw_requires_import": [],
                    },
                },
            }
        }

        config = PlansManager(plans).get_configuration("2x")

        self.assertEqual(config.normalization_schemes, ["ZScoreNormalization"])
        self.assertEqual(config.use_mask_for_norm, [False])
        self.assertTrue(config.batch_dice)

    def test_merge_arch_kwargs_and_derive_patch_size(self):
        plans = {
            "configurations": {
                "2x": {
                    "data_identifier": "base",
                    "preprocessor_name": "DefaultPreprocessor",
                    "batch_size": 2,
                    "patch_size": [16, 32, 48],
                    "patch_size_unit": [16, 32, 48],
                    "patch_size_multiplier": None,
                    "architecture": {
                        "network_class_name": "ParentNet",
                        "arch_kwargs": {
                            "stem": {"stride": [2, 2, 2], "kernel_size": [3, 3, 3]},
                            "features_per_stage": None,
                            "encoder_n_blocks_per_stage": None,
                            "decoder_n_blocks_per_stage": None,
                        },
                        "_kw_requires_import": [],
                    },
                    "required_for_training": [
                        "patch_size_multiplier",
                        "architecture.arch_kwargs.features_per_stage",
                        "architecture.arch_kwargs.encoder_n_blocks_per_stage",
                        "architecture.arch_kwargs.decoder_n_blocks_per_stage",
                    ],
                },
                "child": {
                    "inherits_from": "2x",
                    "patch_size_multiplier": 3,
                    "architecture": {
                        "merge_arch_kwargs": True,
                        "arch_kwargs": {
                            "features_per_stage": [16, 32],
                            "encoder_n_blocks_per_stage": [1, 1],
                            "decoder_n_blocks_per_stage": [1],
                        },
                    },
                },
            }
        }

        config = PlansManager(plans).get_configuration("child")

        self.assertEqual(config.patch_size, [48, 96, 144])
        self.assertEqual(config.network_arch_class_name, "ParentNet")
        self.assertEqual(config.network_arch_init_kwargs["stem"]["stride"], [2, 2, 2])
        self.assertEqual(config.network_arch_init_kwargs["features_per_stage"], [16, 32])
        config.validate_required_for_training("child")

    def test_incomplete_base_configuration_has_helpful_error(self):
        config = PlansManager({
            "configurations": {
                "2x": {
                    "data_identifier": "base",
                    "preprocessor_name": "DefaultPreprocessor",
                    "batch_size": 2,
                    "patch_size": [16, 32, 48],
                    "patch_size_unit": [16, 32, 48],
                    "patch_size_multiplier": None,
                    "architecture": {
                        "network_class_name": "ParentNet",
                        "arch_kwargs": {"features_per_stage": None},
                        "_kw_requires_import": [],
                    },
                    "required_for_training": [
                        "patch_size_multiplier",
                        "architecture.arch_kwargs.features_per_stage",
                    ],
                },
            }
        }).get_configuration("2x")

        with self.assertRaisesRegex(RuntimeError, "patch_size_multiplier"):
            config.validate_required_for_training("2x")


class TestAdamWTrainerConfig(unittest.TestCase):
    def test_applies_trainer_configuration(self):
        trainer = nnUNetTrainerAdamW.__new__(nnUNetTrainerAdamW)
        trainer.configuration_manager = type(
            "Config",
            (),
            {
                "trainer": {
                    "initial_lr": 1e-4,
                    "weight_decay": 0.01,
                    "num_epochs": 100,
                    "warmup_epochs": 10,
                    "min_lr": 1e-7,
                    "enable_deep_supervision": True,
                }
            },
        )()
        trainer.initial_lr = 3e-4
        trainer.weight_decay = 1e-3
        trainer.num_epochs = 250
        trainer.warmup_epochs = 5
        trainer.min_lr = 1e-6
        trainer.enable_deep_supervision = False

        trainer._apply_trainer_configuration()

        self.assertEqual(trainer.initial_lr, 1e-4)
        self.assertEqual(trainer.weight_decay, 0.01)
        self.assertEqual(trainer.num_epochs, 100)
        self.assertEqual(trainer.warmup_epochs, 10)
        self.assertEqual(trainer.min_lr, 1e-7)
        self.assertTrue(trainer.enable_deep_supervision)

    def test_rejects_invalid_schedule_bounds(self):
        trainer = nnUNetTrainerAdamW.__new__(nnUNetTrainerAdamW)
        trainer.configuration_manager = type("Config", (), {"trainer": {"warmup_epochs": 1}})()
        trainer.initial_lr = 3e-4
        trainer.weight_decay = 1e-3
        trainer.num_epochs = 250
        trainer.warmup_epochs = 5
        trainer.min_lr = 1e-6
        trainer.enable_deep_supervision = False

        with self.assertRaisesRegex(ValueError, "warmup_epochs"):
            trainer._apply_trainer_configuration()


if __name__ == "__main__":
    unittest.main()
