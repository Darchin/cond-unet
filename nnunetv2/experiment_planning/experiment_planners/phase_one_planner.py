from nnunetv2.experiment_planning.experiment_planners.extended_stats_planner import ExtendedExperimentPlanner

class PhaseOnePlanner(ExtendedExperimentPlanner):
    def plan_experiment(self):
        super().plan_experiment()
        
        plans = super().plan_experiment()
        
        plans['configurations'] = {
            "base": {
                "preprocessor_name": "DefaultPreprocessor",
                "batch_size": 2,
                "normalization_schemes": [
                    "CTNormalization"
                ],
                "use_mask_for_norm": [
                    False
                ],
                "resampling_fn_data": "resample_data_or_seg_to_shape",
                "resampling_fn_seg": "resample_data_or_seg_to_shape",
                "resampling_fn_data_kwargs": {
                    "is_seg": False,
                    "order": 3,
                    "order_z": 0,
                    "force_separate_z": None
                },
                "resampling_fn_seg_kwargs": {
                    "is_seg": True,
                    "order": 1,
                    "order_z": 0,
                    "force_separate_z": None
                },
                "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
                "resampling_fn_probabilities_kwargs": {
                    "is_seg": False,
                    "order": 1,
                    "order_z": 0,
                    "force_separate_z": None
                },
                "batch_dice": True
            },
            "2x": {
                "inherits_from": "base",
                "data_identifier": "96-192-192_1.50-0.75-0.75",
                "patch_size": [96, 192, 192],
                "spacing": [1.5, 0.75, 0.75]
            },
            "3x": {
                "inherits_from": "base",
                "data_identifier": "64-192-192_2.25-0.75-0.75",
                "patch_size": [64, 192, 192],
                "spacing": [2.25, 0.75, 0.75]
            },
            "4x": {
                "inherits_from": "base",
                "data_identifier": "48-192-192_3.00-0.75-0.75",
                "patch_size": [48, 192, 192],
                "spacing": [3.0, 0.75, 0.75]
            },
            "1-1_S-2X": {
                "inherits_from": "2x",
                "architecture": {
                    "network_class_name": "nnunetv2.training.network_architecture.cond_unet.CondUNet",
                    "arch_kwargs": {
                        "stem_kernel_size": [3, 3, 3],
                        "stem_stride": [1, 2, 2],
                        "n_stages": 5,
                        "features_per_stage": [32, 64, 128, 256, 512],
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        "n_blocks_per_stage": [2, 3, 3, 9, 3],
                        "n_conv_per_stage_decoder": [1, 1, 1, 1],
                        "kernel_sizes": [3, 3, 3, 3, 3],
                        "conv_bias": True,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                        "dropout_op": None,
                        "dropout_op_kwargs": None,
                        "nonlin": "torch.nn.ReLU",
                        "nonlin_kwargs": {"inplace": True},
                        "encoder_expansion_ratio": [2.0, 2.0, 4.0, 4.0, 4.0],
                        "decoder_expansion_ratio": 1.0
                    },
                    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]
                }
            },
            "1-2_M-2X": {
                "inherits_from": "2x",
                "architecture": {
                    "network_class_name": "nnunetv2.training.network_architecture.cond_unet.CondUNet",
                    "arch_kwargs": {
                        "stem_kernel_size": [3, 3, 3],
                        "stem_stride": [1, 2, 2],
                        "n_stages": 5,
                        "features_per_stage": [48, 96, 192, 384, 768],
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        "n_blocks_per_stage": [2, 3, 3, 9, 3],
                        "n_conv_per_stage_decoder": [1, 1, 1, 1],
                        "kernel_sizes": [3, 3, 3, 3, 3],
                        "conv_bias": True,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                        "dropout_op": None,
                        "dropout_op_kwargs": None,
                        "nonlin": "torch.nn.ReLU",
                        "nonlin_kwargs": {"inplace": True},
                        "encoder_expansion_ratio": [2.0, 2.0, 4.0, 4.0, 4.0],
                        "decoder_expansion_ratio": 1.0
                    },
                    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]
                }
            },
            "1-3_L-2X": {
                "inherits_from": "2x",
                "architecture": {
                    "network_class_name": "nnunetv2.training.network_architecture.cond_unet.CondUNet",
                    "arch_kwargs": {
                        "stem_kernel_size": [3, 3, 3],
                        "stem_stride": [1, 2, 2],
                        "n_stages": 5,
                        "features_per_stage": [64, 128, 256, 512, 1024],
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        "n_blocks_per_stage": [2, 3, 3, 9, 3],
                        "n_conv_per_stage_decoder": [1, 1, 1, 1],
                        "kernel_sizes": [3, 3, 3, 3, 3],
                        "conv_bias": True,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                        "dropout_op": None,
                        "dropout_op_kwargs": None,
                        "nonlin": "torch.nn.ReLU",
                        "nonlin_kwargs": {"inplace": True},
                        "encoder_expansion_ratio": [2.0, 2.0, 4.0, 4.0, 4.0],
                        "decoder_expansion_ratio": 1.0
                    },
                    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]
                }
            },
            "1-4_S-3X": {
                "inherits_from": "3x",
                "architecture": {
                    "network_class_name": "nnunetv2.training.network_architecture.cond_unet.CondUNet",
                    "arch_kwargs": {
                        "stem_kernel_size": [3, 5, 5],
                        "stem_stride": [1, 3, 3],
                        "n_stages": 4,
                        "features_per_stage": [48, 96, 192, 384],
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        "n_blocks_per_stage": [2, 3, 9, 3],
                        "n_conv_per_stage_decoder": [1, 1, 1],
                        "kernel_sizes": [3, 3, 3, 3],
                        "conv_bias": True,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                        "dropout_op": None,
                        "dropout_op_kwargs": None,
                        "nonlin": "torch.nn.ReLU",
                        "nonlin_kwargs": {"inplace": True},
                        "encoder_expansion_ratio": [2.0, 2.0, 4.0, 4.0],
                        "decoder_expansion_ratio": 1.0
                    },
                    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]
                }
            },
            "1-5_M-3X": {
                "inherits_from": "3x",
                "architecture": {
                    "network_class_name": "nnunetv2.training.network_architecture.cond_unet.CondUNet",
                    "arch_kwargs": {
                        "stem_kernel_size": [3, 5, 5],
                        "stem_stride": [1, 3, 3],
                        "n_stages": 4,
                        "features_per_stage": [64, 128, 256, 512],
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        "n_blocks_per_stage": [2, 3, 9, 3],
                        "n_conv_per_stage_decoder": [1, 1, 1],
                        "kernel_sizes": [3, 3, 3, 3],
                        "conv_bias": True,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                        "dropout_op": None,
                        "dropout_op_kwargs": None,
                        "nonlin": "torch.nn.ReLU",
                        "nonlin_kwargs": {"inplace": True},
                        "encoder_expansion_ratio": [2.0, 2.0, 4.0, 4.0],
                        "decoder_expansion_ratio": 1.0
                    },
                    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]
                }
            },
            "1-6_S-3X": {
                "inherits_from": "3x",
                "architecture": {
                    "network_class_name": "nnunetv2.training.network_architecture.cond_unet.CondUNet",
                    "arch_kwargs": {
                        "stem_kernel_size": [3, 5, 5],
                        "stem_stride": [1, 3, 3],
                        "n_stages": 4,
                        "features_per_stage": [96, 192, 384, 768],
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        "n_blocks_per_stage": [2, 3, 9, 3],
                        "n_conv_per_stage_decoder": [1, 1, 1],
                        "kernel_sizes": [3, 3, 3, 3],
                        "conv_bias": True,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                        "dropout_op": None,
                        "dropout_op_kwargs": None,
                        "nonlin": "torch.nn.ReLU",
                        "nonlin_kwargs": {"inplace": True},
                        "encoder_expansion_ratio": [2.0, 2.0, 4.0, 4.0],
                        "decoder_expansion_ratio": 1.0
                    },
                    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]
                }
            },
            "1-7_S-4X": {
                "inherits_from": "4x",
                "architecture": {
                    "network_class_name": "nnunetv2.training.network_architecture.cond_unet.CondUNet",
                    "arch_kwargs": {
                        "stem_kernel_size": [3, 7, 7],
                        "stem_stride": [1, 4, 4],
                        "n_stages": 4,
                        "features_per_stage": [64, 128, 256, 512],
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        "n_blocks_per_stage": [3, 3, 9, 3],
                        "n_conv_per_stage_decoder": [1, 1, 1],
                        "kernel_sizes": [3, 3, 3, 3],
                        "conv_bias": True,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                        "dropout_op": None,
                        "dropout_op_kwargs": None,
                        "nonlin": "torch.nn.ReLU",
                        "nonlin_kwargs": {"inplace": True},
                        "encoder_expansion_ratio": [2.0, 4.0, 4.0, 4.0],
                        "decoder_expansion_ratio": 1.0
                    },
                    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]
                }
            },
            "1-8_M-4X": {
                "inherits_from": "4x",
                "architecture": {
                    "network_class_name": "nnunetv2.training.network_architecture.cond_unet.CondUNet",
                    "arch_kwargs": {
                        "stem_kernel_size": [3, 7, 7],
                        "stem_stride": [1, 4, 4],
                        "n_stages": 4,
                        "features_per_stage": [96, 192, 384, 768],
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        "n_blocks_per_stage": [3, 3, 9, 3],
                        "n_conv_per_stage_decoder": [1, 1, 1],
                        "kernel_sizes": [3, 3, 3, 3],
                        "conv_bias": True,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                        "dropout_op": None,
                        "dropout_op_kwargs": None,
                        "nonlin": "torch.nn.ReLU",
                        "nonlin_kwargs": {"inplace": True},
                        "encoder_expansion_ratio": [2.0, 4.0, 4.0, 4.0],
                        "decoder_expansion_ratio": 1.0
                    },
                    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]
                }
            },
            "1-9_L-4X": {
                "inherits_from": "4x",
                "architecture": {
                    "network_class_name": "nnunetv2.training.network_architecture.cond_unet.CondUNet",
                    "arch_kwargs": {
                        "stem_kernel_size": [3, 7, 7],
                        "stem_stride": [1, 4, 4],
                        "n_stages": 4,
                        "features_per_stage": [128, 256, 512, 1024],
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        "n_blocks_per_stage": [3, 3, 9, 3],
                        "n_conv_per_stage_decoder": [1, 1, 1],
                        "kernel_sizes": [3, 3, 3, 3],
                        "conv_bias": True,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                        "dropout_op": None,
                        "dropout_op_kwargs": None,
                        "nonlin": "torch.nn.ReLU",
                        "nonlin_kwargs": {"inplace": True},
                        "encoder_expansion_ratio": [2.0, 4.0, 4.0, 4.0],
                        "decoder_expansion_ratio": 1.0
                    },
                    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]
                }
            }
        }

        self.save_plans(plans)
        return plans