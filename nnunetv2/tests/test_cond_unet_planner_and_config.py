import unittest

from nnunetv2.experiment_planning.dataset_fingerprint.cond_unet_fingerprint_extractor import (
    CondUNetFingerprintExtractor,
)
from nnunetv2.experiment_planning.experiment_planners.cond_unet_planner import (
    CondUNetPlanner,
    PhaseOnePlanner,
    PhaseThreePlanner,
    PhaseTwoPlanner,
)
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerAdamW import nnUNetTrainerAdamW
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


class TestCondUNetPlannerHelpers(unittest.TestCase):
    @staticmethod
    def _minimal_planner(planner_class=CondUNetPlanner):
        planner = planner_class.__new__(planner_class)
        planner.dataset_fingerprint = {
            "spacing_percentiles": {
                "50": [1.0, 1.0, 1.0],
                "45": [1.0, 1.0, 1.0],
                "40": [1.0, 1.0, 1.0],
                "35": [1.0, 1.0, 1.0],
                "30": [1.0, 1.0, 1.0],
                "25": [1.0, 1.0, 1.0],
            },
            "spacings": [[1.0, 1.0, 1.0]],
            "shapes_after_crop": [[128, 128, 128]],
        }
        planner.overwrite_target_spacing = None
        planner.preprocessor_name = "DefaultPreprocessor"
        planner.generate_data_identifier = lambda configuration_name: f"data_{configuration_name}"
        return planner

    def test_presets_use_expected_stage_depths(self):
        self.assertEqual(CondUNetPlanner.presets["2x"]["post_stem_downsampling_stages"], 4)
        self.assertEqual(CondUNetPlanner.presets["3x"]["post_stem_downsampling_stages"], 3)
        self.assertEqual(CondUNetPlanner.presets["4x"]["post_stem_downsampling_stages"], 3)

    def test_presets_use_expected_architecture_defaults(self):
        self.assertEqual(
            CondUNetPlanner.presets["2x"]["arch_kwargs"],
            {
                "encoder_n_blocks_per_stage": [2, 3, 3, 9, 3],
                "decoder_n_blocks_per_stage": [1, 1, 1, 1],
                "encoder_expansion_ratio": [2.0, 2.0, 4.0, 4.0, 4.0],
                "decoder_expansion_ratio": 1.0,
            },
        )
        for configuration_name in ("3x", "4x"):
            self.assertEqual(
                CondUNetPlanner.presets[configuration_name]["arch_kwargs"],
                {
                    "encoder_n_blocks_per_stage": [3, 3, 9, 3],
                    "decoder_n_blocks_per_stage": [1, 1, 1],
                    "encoder_expansion_ratio": [2.0, 4.0, 4.0, 4.0],
                    "decoder_expansion_ratio": 1.0,
                },
            )

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

    def test_stem_stride_can_make_axis_finer_than_reference_post_stem_spacing(self):
        stem_stride = CondUNetPlanner.determine_stem_stride(
            median_spacing=[1.0, 2.0, 1.0],
            target_spacing=[1.0, 1.3, 1.0],
            stem_factor=3,
        )

        self.assertEqual(stem_stride.tolist(), [3, 2, 3])

    def test_patch_geometry_uses_physical_aspect_ratio_and_total_stride(self):
        aspect_ratio, patch_size_unit = CondUNetPlanner.compute_patch_geometry(
            target_spacing=[4.0, 1.0, 1.0],
            median_shape=[10, 20, 40],
            stem_stride=[1, 3, 3],
            post_stem_downsampling_stages=4,
        )

        self.assertEqual(aspect_ratio, [2, 1, 2])
        self.assertEqual(patch_size_unit, [32, 48, 96])

    def test_patch_size_unit_mm_multiplies_patch_unit_by_spacing(self):
        patch_size_unit_mm = CondUNetPlanner.compute_patch_size_unit_mm(
            patch_size_unit=[32, 48, 96],
            target_spacing=[4.0, 1.0, 1.5],
        )

        self.assertEqual(patch_size_unit_mm, [128.0, 48.0, 144.0])

    def test_stem_kernel_size_is_clamped_to_minimum_three(self):
        stem_kernel_size = CondUNetPlanner.compute_stem_kernel_size([1, 2, 3, 4])

        self.assertEqual(stem_kernel_size, [3, 3, 5, 7])

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

    def test_preset_plan_applies_architecture_defaults(self):
        planner = self._minimal_planner()

        plan = planner._plan_for_preset("2x", [0, 1, 2])

        self.assertEqual(plan["architecture"]["arch_kwargs"]["encoder_n_blocks_per_stage"], [2, 3, 3, 9, 3])
        self.assertEqual(plan["architecture"]["arch_kwargs"]["decoder_n_blocks_per_stage"], [1, 1, 1, 1])
        self.assertEqual(plan["architecture"]["arch_kwargs"]["encoder_expansion_ratio"], [2.0, 2.0, 4.0, 4.0, 4.0])
        self.assertEqual(plan["architecture"]["arch_kwargs"]["decoder_expansion_ratio"], 1.0)
        self.assertEqual(
            plan["required_for_training"],
            [
                "patch_size_multiplier",
                "architecture.arch_kwargs.features_per_stage",
            ],
        )

    def test_phase_one_planner_generates_expected_child_configurations(self):
        planner = PhaseOnePlanner.__new__(PhaseOnePlanner)

        configurations = planner._additional_configurations()

        self.assertEqual(
            list(configurations),
            [
                "2x-s", "2x-m", "2x-l",
                "3x-s", "3x-m", "3x-l",
                "4x-s", "4x-m", "4x-l",
            ],
        )
        self.assertEqual(configurations["2x-s"]["inherits_from"], "2x")
        self.assertEqual(configurations["2x-s"]["patch_size_multiplier"], 6)
        self.assertEqual(
            configurations["2x-s"]["architecture"]["arch_kwargs"]["features_per_stage"],
            [32, 64, 128, 256, 512],
        )
        self.assertEqual(configurations["3x-l"]["inherits_from"], "3x")
        self.assertEqual(configurations["3x-l"]["patch_size_multiplier"], 8)
        self.assertEqual(
            configurations["3x-l"]["architecture"]["arch_kwargs"]["features_per_stage"],
            [96, 192, 384, 768],
        )
        self.assertEqual(configurations["4x-l"]["inherits_from"], "4x")
        self.assertEqual(configurations["4x-l"]["patch_size_multiplier"], 6)
        self.assertEqual(
            configurations["4x-l"]["architecture"]["arch_kwargs"]["features_per_stage"],
            [128, 256, 512, 1024],
        )

    def test_phase_one_child_configuration_resolves_as_trainable(self):
        planner = self._minimal_planner(PhaseOnePlanner)
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
                "2x": planner._plan_for_preset("2x", [0, 1, 2]),
                "2x-m": planner._additional_configurations()["2x-m"],
            },
        }

        config = PlansManager(plans).get_configuration("2x-m")

        self.assertEqual(config.patch_size, [192, 192, 192])
        self.assertEqual(config.patch_size_multiplier, 6)
        self.assertEqual(config.network_arch_init_kwargs["features_per_stage"], [48, 96, 192, 384, 768])
        self.assertEqual(config.network_arch_init_kwargs["encoder_n_blocks_per_stage"], [2, 3, 3, 9, 3])
        self.assertEqual(config.network_arch_init_kwargs["decoder_n_blocks_per_stage"], [1, 1, 1, 1])
        self.assertEqual(config.network_arch_init_kwargs["encoder_expansion_ratio"], [2.0, 2.0, 4.0, 4.0, 4.0])
        self.assertEqual(config.network_arch_init_kwargs["decoder_expansion_ratio"], 1.0)
        config.validate_required_for_training("2x-m")

    def test_phase_two_planner_generates_expected_child_configurations(self):
        planner = PhaseTwoPlanner.__new__(PhaseTwoPlanner)

        configurations = planner._additional_configurations()

        self.assertEqual(
            list(configurations),
            [
                "4x-m",
                "4x-m-gse-enck",
                "4x-m-gse-deck",
                "4x-m-gse-enck-deck",
                "4x-m-gcc-enck",
                "4x-m-gcc-deck",
                "4x-m-gcc-enck-deck",
            ],
        )
        self.assertEqual(configurations["4x-m"]["inherits_from"], "4x")
        self.assertEqual(configurations["4x-m"]["patch_size_multiplier"], 6)
        self.assertEqual(
            configurations["4x-m"]["architecture"]["arch_kwargs"]["features_per_stage"],
            [96, 192, 384, 768],
        )
        self.assertEqual(configurations["4x-m-gse-enck"]["inherits_from"], "4x-m")
        self.assertEqual(
            configurations["4x-m-gse-enck"]["architecture"]["arch_kwargs"]["se"],
            {"encoder": [False, True, True, True]},
        )
        self.assertEqual(
            configurations["4x-m-gse-deck"]["architecture"]["arch_kwargs"]["se"],
            {"decoder": [False, True, True]},
        )
        self.assertEqual(
            configurations["4x-m-gse-enck-deck"]["architecture"]["arch_kwargs"]["se"],
            {
                "encoder": [False, True, True, True],
                "decoder": [False, True, True],
            },
        )
        self.assertEqual(
            configurations["4x-m-gcc-enck"]["architecture"]["arch_kwargs"]["cc"],
            {
                "encoder": [False, True, True, True],
                "encoder_num_experts": 4,
            },
        )
        self.assertEqual(
            configurations["4x-m-gcc-deck"]["architecture"]["arch_kwargs"]["cc"],
            {
                "decoder": [False, True, True],
                "decoder_num_experts": 4,
            },
        )
        self.assertEqual(
            configurations["4x-m-gcc-enck-deck"]["architecture"]["arch_kwargs"]["cc"],
            {
                "encoder": [False, True, True, True],
                "encoder_num_experts": 4,
                "decoder": [False, True, True],
                "decoder_num_experts": 4,
            },
        )

    def test_phase_two_child_configuration_resolves_as_trainable(self):
        planner = self._minimal_planner(PhaseTwoPlanner)
        phase_two_configurations = planner._additional_configurations()
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
                "4x": planner._plan_for_preset("4x", [0, 1, 2]),
                "4x-m": phase_two_configurations["4x-m"],
                "4x-m-gse-enck-deck": phase_two_configurations["4x-m-gse-enck-deck"],
                "4x-m-gcc-enck-deck": phase_two_configurations["4x-m-gcc-enck-deck"],
            },
        }

        se_config = PlansManager(plans).get_configuration("4x-m-gse-enck-deck")
        cc_config = PlansManager(plans).get_configuration("4x-m-gcc-enck-deck")

        for config, configuration_name in (
            (se_config, "4x-m-gse-enck-deck"),
            (cc_config, "4x-m-gcc-enck-deck"),
        ):
            self.assertEqual(config.patch_size, [192, 192, 192])
            self.assertEqual(config.patch_size_multiplier, 6)
            self.assertEqual(config.network_arch_init_kwargs["features_per_stage"], [96, 192, 384, 768])
            self.assertEqual(config.network_arch_init_kwargs["encoder_n_blocks_per_stage"], [3, 3, 9, 3])
            self.assertEqual(config.network_arch_init_kwargs["decoder_n_blocks_per_stage"], [1, 1, 1])
            config.validate_required_for_training(configuration_name)

        self.assertEqual(
            se_config.network_arch_init_kwargs["se"],
            {
                "encoder": [False, True, True, True],
                "decoder": [False, True, True],
            },
        )
        self.assertEqual(
            cc_config.network_arch_init_kwargs["cc"],
            {
                "encoder": [False, True, True, True],
                "encoder_num_experts": 4,
                "decoder": [False, True, True],
                "decoder_num_experts": 4,
            },
        )

    def test_phase_three_planner_generates_expected_child_configurations(self):
        planner = PhaseThreePlanner.__new__(PhaseThreePlanner)

        configurations = planner._additional_configurations()

        self.assertEqual(
            list(configurations),
            ["4x-m", "4x-m-t4se-enck", "4x-m-t4cc-enck"],
        )
        self.assertEqual(configurations["4x-m"]["inherits_from"], "4x")
        self.assertEqual(configurations["4x-m"]["patch_size_multiplier"], 6)
        self.assertEqual(
            configurations["4x-m"]["architecture"]["arch_kwargs"]["features_per_stage"],
            [96, 192, 384, 768],
        )
        self.assertEqual(configurations["4x-m-t4se-enck"]["inherits_from"], "4x-m")
        self.assertEqual(
            configurations["4x-m-t4se-enck"]["architecture"]["arch_kwargs"]["se"],
            {
                "encoder": [False, True, True, True],
                "tile_size": [16, 48, 48],
            },
        )
        self.assertEqual(configurations["4x-m-t4cc-enck"]["inherits_from"], "4x-m")
        self.assertEqual(
            configurations["4x-m-t4cc-enck"]["architecture"]["arch_kwargs"]["cc"],
            {
                "encoder": [False, True, True, True],
                "encoder_num_experts": 4,
                "tile_size": [16, 48, 48],
            },
        )

    def test_phase_three_child_configurations_resolve_as_trainable(self):
        planner = self._minimal_planner(PhaseThreePlanner)
        phase_three_configurations = planner._additional_configurations()
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
                "4x": planner._plan_for_preset("4x", [0, 1, 2]),
                "4x-m": phase_three_configurations["4x-m"],
                "4x-m-t4se-enck": phase_three_configurations["4x-m-t4se-enck"],
                "4x-m-t4cc-enck": phase_three_configurations["4x-m-t4cc-enck"],
            },
        }

        se_config = PlansManager(plans).get_configuration("4x-m-t4se-enck")
        cc_config = PlansManager(plans).get_configuration("4x-m-t4cc-enck")

        for config, configuration_name in (
            (se_config, "4x-m-t4se-enck"),
            (cc_config, "4x-m-t4cc-enck"),
        ):
            self.assertEqual(config.patch_size, [192, 192, 192])
            self.assertEqual(config.patch_size_multiplier, 6)
            self.assertEqual(config.network_arch_init_kwargs["features_per_stage"], [96, 192, 384, 768])
            self.assertEqual(config.network_arch_init_kwargs["encoder_n_blocks_per_stage"], [3, 3, 9, 3])
            self.assertEqual(config.network_arch_init_kwargs["decoder_n_blocks_per_stage"], [1, 1, 1])
            config.validate_required_for_training(configuration_name)

        self.assertEqual(
            se_config.network_arch_init_kwargs["se"],
            {
                "encoder": [False, True, True, True],
                "tile_size": [16, 48, 48],
            },
        )
        self.assertEqual(
            cc_config.network_arch_init_kwargs["cc"],
            {
                "encoder": [False, True, True, True],
                "encoder_num_experts": 4,
                "tile_size": [16, 48, 48],
            },
        )


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
                    "patch_size_unit_mm": [16.0, 32.0, 48.0],
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

    def test_nested_override_merges_child_values_and_derives_patch_size(self):
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
                    "trainer": {
                        "initial_lr": 0.01,
                        "weight_decay": 0.001,
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
                        "arch_kwargs": {
                            "features_per_stage": [16, 32],
                            "encoder_n_blocks_per_stage": [1, 1],
                            "decoder_n_blocks_per_stage": [1],
                        },
                    },
                    "trainer": {
                        "initial_lr": 0.001,
                    },
                },
            }
        }

        config = PlansManager(plans).get_configuration("child")

        self.assertEqual(config.patch_size, [48, 96, 144])
        self.assertEqual(config.network_arch_class_name, "ParentNet")
        self.assertEqual(config.network_arch_init_kwargs["stem"]["stride"], [2, 2, 2])
        self.assertEqual(config.network_arch_init_kwargs["features_per_stage"], [16, 32])
        self.assertEqual(config.trainer, {"initial_lr": 0.001, "weight_decay": 0.001})
        config.validate_required_for_training("child")

    def test_nested_override_false_replaces_top_level_child_dicts(self):
        plans = {
            "configurations": {
                "parent": {
                    "architecture": {
                        "network_class_name": "ParentNet",
                        "arch_kwargs": {
                            "stem": {"stride": [2, 2, 2], "kernel_size": [3, 3, 3]},
                            "features_per_stage": [8, 16],
                        },
                        "_kw_requires_import": [],
                    },
                    "trainer": {
                        "initial_lr": 0.01,
                        "weight_decay": 0.001,
                    },
                },
                "child": {
                    "inherits_from": "parent",
                    "nested_override": False,
                    "architecture": {
                        "arch_kwargs": {
                            "features_per_stage": [16, 32],
                        },
                    },
                    "trainer": {
                        "initial_lr": 0.001,
                    },
                },
            }
        }

        config = PlansManager(plans)._internal_resolve_configuration_inheritance("child")

        self.assertEqual(config["architecture"], {"arch_kwargs": {"features_per_stage": [16, 32]}})
        self.assertEqual(config["trainer"], {"initial_lr": 0.001})

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
