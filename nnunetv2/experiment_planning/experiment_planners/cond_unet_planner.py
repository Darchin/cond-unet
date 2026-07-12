import shutil
from typing import Dict, List, Tuple, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from dynamic_network_architectures.building_blocks.helper import (
    convert_dim_to_conv_op,
    get_matching_instancenorm,
)

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import (
    ExperimentPlanner,
)
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape


class CondUNetPlanner(ExperimentPlanner):
    presets = {
        "2x": {
            "stem_factor": 2,
            "post_stem_downsampling_stages": 4,
            "arch_kwargs": {
                "encoder_n_blocks_per_stage": [2, 3, 3, 9, 3],
                "decoder_n_blocks_per_stage": [1, 1, 1, 1],
                "encoder_expansion_ratio": [2.0, 2.0, 4.0, 4.0, 4.0],
                "decoder_expansion_ratio": 1.0,
            },
        },
        "3x": {
            "stem_factor": 3,
            "post_stem_downsampling_stages": 3,
            "arch_kwargs": {
                "encoder_n_blocks_per_stage": [3, 3, 9, 3],
                "decoder_n_blocks_per_stage": [1, 1, 1],
                "encoder_expansion_ratio": [2.0, 4.0, 4.0, 4.0],
                "decoder_expansion_ratio": 1.0,
            },
        },
        "4x": {
            "stem_factor": 4,
            "post_stem_downsampling_stages": 3,
            "arch_kwargs": {
                "encoder_n_blocks_per_stage": [3, 3, 9, 3],
                "decoder_n_blocks_per_stage": [1, 1, 1],
                "encoder_expansion_ratio": [2.0, 4.0, 4.0, 4.0],
                "decoder_expansion_ratio": 1.0,
            },
        },
    }
    spacing_percentile_min = 25
    spacing_percentile_step = 5
    UNet_min_batch_size = 2
    trainer_defaults = {
        "initial_lr": 3e-4,
        "weight_decay": 1e-3,
        "num_epochs": 250,
        "warmup_epochs": 5,
        "min_lr": 1e-6,
        "enable_deep_supervision": False,
    }

    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        gpu_memory_target_in_gb: float = 8,
        preprocessor_name: str = "DefaultPreprocessor",
        plans_name: str = "nnUNetCondUNetPlans",
        overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
        suppress_transpose: bool = False,
    ):
        super().__init__(
            dataset_name_or_id,
            gpu_memory_target_in_gb,
            preprocessor_name,
            plans_name,
            overwrite_target_spacing,
            suppress_transpose,
        )

    @staticmethod
    def _percentile_spacing(
        spacing_percentiles: Dict[str, List[float]], percentile: int
    ) -> np.ndarray:
        key = str(int(percentile))
        if key not in spacing_percentiles:
            raise RuntimeError(
                f"spacing_percentiles is missing percentile {key}. Rerun fingerprint extraction with "
                "-fpe CondUNetFingerprintExtractor --clean."
            )
        return np.asarray(spacing_percentiles[key], dtype=float)

    @classmethod
    def determine_target_spacing_for_factor(
        cls,
        spacing_percentiles: Dict[str, List[float]],
        stem_factor: int,
        percentile_min: int = None,
        percentile_step: int = None,
    ) -> np.ndarray:
        percentile_min = (
            cls.spacing_percentile_min if percentile_min is None else percentile_min
        )
        percentile_step = (
            cls.spacing_percentile_step if percentile_step is None else percentile_step
        )
        median_spacing = cls._percentile_spacing(spacing_percentiles, 50)
        reference_spacing = float(np.min(median_spacing))
        target_spacing = median_spacing.copy()

        for axis in range(len(median_spacing)):
            if median_spacing[axis] / reference_spacing <= stem_factor:
                continue

            previous_spacing = median_spacing[axis]
            selected_spacing = None
            for percentile in range(
                50 - percentile_step, percentile_min - 1, -percentile_step
            ):
                candidate_spacing = cls._percentile_spacing(
                    spacing_percentiles, percentile
                )[axis]
                if candidate_spacing / reference_spacing < stem_factor:
                    selected_spacing = previous_spacing
                    break
                previous_spacing = candidate_spacing

            target_spacing[axis] = (
                cls._percentile_spacing(spacing_percentiles, percentile_min)[axis]
                if selected_spacing is None
                else selected_spacing
            )

        return target_spacing

    @staticmethod
    def determine_stem_stride(
        median_spacing: np.ndarray, target_spacing: np.ndarray, stem_factor: int
    ) -> np.ndarray:
        median_spacing = np.asarray(median_spacing, dtype=float)
        target_spacing = np.asarray(target_spacing, dtype=float)
        reference_spacing = float(np.min(median_spacing))
        reference_axes = np.isclose(median_spacing, reference_spacing)
        reference_post_stem_spacing = reference_spacing * stem_factor
        stride = np.ones_like(target_spacing, dtype=int)
        stride_candidates = np.arange(1, stem_factor + 1, dtype=int)

        for axis in range(len(target_spacing)):
            post_stem_spacing = target_spacing[axis] * stride_candidates
            anisotropy = np.maximum(
                post_stem_spacing / reference_post_stem_spacing,
                reference_post_stem_spacing / post_stem_spacing,
            )
            stride[axis] = int(stride_candidates[np.argmin(anisotropy)])

        stride[reference_axes] = stem_factor
        return stride

    @staticmethod
    def compute_patch_geometry(
        target_spacing: np.ndarray,
        median_shape: np.ndarray,
        stem_stride: np.ndarray,
        post_stem_downsampling_stages: int,
    ) -> Tuple[List[int], List[int]]:
        physical_extent = np.asarray(target_spacing, dtype=float) * np.asarray(
            median_shape, dtype=float
        )
        aspect_ratio = np.maximum(
            1, np.rint(physical_extent / np.min(physical_extent)).astype(int)
        )
        post_stem_stride = np.asarray(
            [2**post_stem_downsampling_stages] * len(stem_stride), dtype=int
        )
        patch_size_unit = (
            np.asarray(stem_stride, dtype=int) * post_stem_stride * aspect_ratio
        ).astype(int)
        return [int(i) for i in aspect_ratio], [int(i) for i in patch_size_unit]

    @staticmethod
    def compute_patch_size_unit_mm(
        patch_size_unit: List[int], target_spacing: np.ndarray
    ) -> List[float]:
        patch_size_unit_mm = np.asarray(patch_size_unit, dtype=float) * np.asarray(
            target_spacing, dtype=float
        )
        return [float(i) for i in patch_size_unit_mm]

    @staticmethod
    def compute_stem_kernel_size(stem_stride: List[int]) -> List[int]:
        return [max(3, int(2 * i - 1)) for i in stem_stride]

    @staticmethod
    def _transpose(
        spatial_values: Union[np.ndarray, List[int], List[float]],
        transpose_forward: List[int],
    ) -> np.ndarray:
        return np.asarray(spatial_values)[transpose_forward]

    def determine_transpose(self):
        if self.suppress_transpose:
            return [0, 1, 2], [0, 1, 2]

        median_spacing = np.median(
            np.vstack(self.dataset_fingerprint["spacings"]), axis=0
        )
        max_spacing_axis = int(np.argmax(median_spacing))
        remaining_axes = [i for i in range(3) if i != max_spacing_axis]
        transpose_forward = [max_spacing_axis] + remaining_axes
        transpose_backward = [
            int(np.argwhere(np.array(transpose_forward) == i)[0][0]) for i in range(3)
        ]
        return transpose_forward, transpose_backward

    def _get_spacing_percentiles(self) -> Dict[str, List[float]]:
        if "spacing_percentiles" not in self.dataset_fingerprint:
            raise RuntimeError(
                "CondUNetPlanner requires spacing_percentiles in dataset_fingerprint.json. "
                "Rerun fingerprint extraction with -fpe CondUNetFingerprintExtractor --clean."
            )
        return self.dataset_fingerprint["spacing_percentiles"]

    def _get_median_shape_at_spacing(
        self, target_spacing: np.ndarray, transpose_forward: List[int]
    ) -> np.ndarray:
        new_shapes = [
            compute_new_shape(shape, spacing, target_spacing)
            for spacing, shape in zip(
                self.dataset_fingerprint["spacings"],
                self.dataset_fingerprint["shapes_after_crop"],
            )
        ]
        return self._transpose(np.median(new_shapes, axis=0), transpose_forward)

    def _architecture(
        self,
        dim: int,
        n_stages: int,
        post_stem_downsampling_stages: int,
        stem_stride: List[int],
        stem_kernel_size: List[int],
        arch_kwargs_defaults: dict = None,
    ) -> dict:
        conv_op = convert_dim_to_conv_op(dim)
        norm = get_matching_instancenorm(conv_op)
        stage_strides = [[1] * dim] + [
            [2] * dim for _ in range(post_stem_downsampling_stages)
        ]
        kernel_sizes = [[3] * dim for _ in range(n_stages)]
        arch_kwargs_defaults = (
            {} if arch_kwargs_defaults is None else dict(arch_kwargs_defaults)
        )
        arch_kwargs = {
            "n_stages": n_stages,
            "features_per_stage": None,
            "conv_op": conv_op.__module__ + "." + conv_op.__name__,
            "kernel_sizes": kernel_sizes,
            "strides": stage_strides,
            "encoder_n_blocks_per_stage": None,
            "decoder_n_blocks_per_stage": None,
            "conv_bias": True,
            "norm_op": norm.__module__ + "." + norm.__name__,
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.ReLU",
            "nonlin_kwargs": {"inplace": True},
            "stem": {
                "stride": stem_stride,
                "kernel_size": stem_kernel_size,
            },
        }
        arch_kwargs.update(arch_kwargs_defaults)
        return {
            "network_class_name": "nnunetv2.training.network_architecture.cond_unet.CondUNet",
            "arch_kwargs": arch_kwargs,
            "_kw_requires_import": ("conv_op", "norm_op", "dropout_op", "nonlin"),
        }

    def _base_configuration(self) -> dict:
        (
            resampling_data,
            resampling_data_kwargs,
            resampling_seg,
            resampling_seg_kwargs,
        ) = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = (
            self.determine_segmentation_softmax_export_fn()
        )
        normalization_schemes, mask_is_used_for_norm = (
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()
        )
        return {
            "normalization_schemes": normalization_schemes,
            "use_mask_for_norm": mask_is_used_for_norm,
            "resampling_fn_data": resampling_data.__name__,
            "resampling_fn_seg": resampling_seg.__name__,
            "resampling_fn_data_kwargs": resampling_data_kwargs,
            "resampling_fn_seg_kwargs": resampling_seg_kwargs,
            "resampling_fn_probabilities": resampling_softmax.__name__,
            "resampling_fn_probabilities_kwargs": resampling_softmax_kwargs,
        }

    def _plan_for_preset(
        self, configuration_name: str, transpose_forward: List[int]
    ) -> dict:
        preset = self.presets[configuration_name]
        stem_factor = preset["stem_factor"]
        post_stem_downsampling_stages = preset["post_stem_downsampling_stages"]
        spacing_percentiles = self._get_spacing_percentiles()
        median_spacing = self._percentile_spacing(spacing_percentiles, 50)
        target_spacing = (
            np.asarray(self.overwrite_target_spacing, dtype=float)
            if self.overwrite_target_spacing is not None
            else self.determine_target_spacing_for_factor(
                spacing_percentiles, stem_factor
            )
        )
        stem_stride = self.determine_stem_stride(
            median_spacing, target_spacing, stem_factor
        )

        target_spacing_transposed = self._transpose(target_spacing, transpose_forward)
        stem_stride_transposed = self._transpose(stem_stride, transpose_forward).astype(
            int
        )
        median_shape_transposed = self._get_median_shape_at_spacing(
            target_spacing, transpose_forward
        )
        aspect_ratio, patch_size_unit = self.compute_patch_geometry(
            target_spacing_transposed,
            median_shape_transposed,
            stem_stride_transposed,
            post_stem_downsampling_stages,
        )
        patch_size_unit_mm = self.compute_patch_size_unit_mm(
            patch_size_unit, target_spacing_transposed
        )
        stem_stride_list = [int(i) for i in stem_stride_transposed]
        stem_kernel_size = self.compute_stem_kernel_size(stem_stride_list)

        n_stages = post_stem_downsampling_stages + 1

        return {
            "inherits_from": "base",
            "data_identifier": self.generate_data_identifier(configuration_name),
            "preprocessor_name": self.preprocessor_name,
            "batch_size": self.UNet_min_batch_size,
            "patch_size": patch_size_unit,
            "patch_size_unit": patch_size_unit,
            "patch_size_unit_mm": patch_size_unit_mm,
            "patch_size_multiplier": None,
            "patch_size_aspect_ratio": aspect_ratio,
            "median_image_size_in_voxels": [
                int(round(i)) for i in median_shape_transposed
            ],
            "spacing": [float(i) for i in target_spacing_transposed],
            "batch_dice": True,
            "architecture": self._architecture(
                len(target_spacing_transposed),
                n_stages,
                post_stem_downsampling_stages,
                stem_stride_list,
                stem_kernel_size,
                preset["arch_kwargs"],
            ),
            "trainer": dict(self.trainer_defaults),
            "required_for_training": [
                "patch_size_multiplier",
                "architecture.arch_kwargs.features_per_stage",
            ],
        }

    def _additional_configurations(self) -> dict:
        return {}

    def plan_experiment(self):
        median_spacing = np.median(
            np.vstack(self.dataset_fingerprint["spacings"]), axis=0
        )
        if len(median_spacing) != 3:
            raise RuntimeError("CondUNetPlanner only generates 3D configurations.")

        transpose_forward, transpose_backward = self.determine_transpose()
        configurations = {"base": self._base_configuration()}
        configurations.update(
            {
                name: self._plan_for_preset(name, transpose_forward)
                for name in self.presets
            }
        )
        configurations.update(self._additional_configurations())

        median_shape = np.median(self.dataset_fingerprint["shapes_after_crop"], axis=0)[
            transpose_forward
        ]
        median_spacing_transposed = median_spacing[transpose_forward]
        preprocessed_dataset_folder = join(nnUNet_preprocessed, self.dataset_name)
        maybe_mkdir_p(preprocessed_dataset_folder)
        shutil.copy(
            join(self.raw_dataset_folder, "dataset.json"),
            join(preprocessed_dataset_folder, "dataset.json"),
        )

        plans = {
            "dataset_name": self.dataset_name,
            "plans_name": self.plans_identifier,
            "original_median_spacing_after_transp": [
                float(i) for i in median_spacing_transposed
            ],
            "original_median_shape_after_transp": [int(round(i)) for i in median_shape],
            "image_reader_writer": self.determine_reader_writer().__name__,
            "transpose_forward": [int(i) for i in transpose_forward],
            "transpose_backward": [int(i) for i in transpose_backward],
            "configurations": configurations,
            "experiment_planner_used": self.__class__.__name__,
            "label_manager": "LabelManager",
            "foreground_intensity_properties_per_channel": self.dataset_fingerprint[
                "foreground_intensity_properties_per_channel"
            ],
        }

        for name, plan in configurations.items():
            print(f"{name} CondUNet base configuration:")
            print(plan)
            print()

        self.plans = plans
        self.save_plans(plans)
        return plans


class PhaseOnePlanner(CondUNetPlanner):
    phase_one_presets = {
        "2x-s": {
            "inherits_from": "2x",
            "patch_size_multiplier": 6,
            "features_per_stage": [32, 64, 128, 256, 512],
        },
        "2x-m": {
            "inherits_from": "2x",
            "patch_size_multiplier": 6,
            "features_per_stage": [48, 96, 192, 384, 768],
        },
        "2x-l": {
            "inherits_from": "2x",
            "patch_size_multiplier": 6,
            "features_per_stage": [64, 128, 256, 512, 1024],
        },
        "3x-s": {
            "inherits_from": "3x",
            "patch_size_multiplier": 8,
            "features_per_stage": [48, 96, 192, 384],
        },
        "3x-m": {
            "inherits_from": "3x",
            "patch_size_multiplier": 8,
            "features_per_stage": [64, 128, 256, 512],
        },
        "3x-l": {
            "inherits_from": "3x",
            "patch_size_multiplier": 8,
            "features_per_stage": [96, 192, 384, 768],
        },
        "4x-s": {
            "inherits_from": "4x",
            "patch_size_multiplier": 6,
            "features_per_stage": [64, 128, 256, 512],
        },
        "4x-m": {
            "inherits_from": "4x",
            "patch_size_multiplier": 6,
            "features_per_stage": [96, 192, 384, 768],
        },
        "4x-l": {
            "inherits_from": "4x",
            "patch_size_multiplier": 6,
            "features_per_stage": [128, 256, 512, 1024],
        },
    }

    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        gpu_memory_target_in_gb: float = 8,
        preprocessor_name: str = "DefaultPreprocessor",
        plans_name: str = "nnUNetCondUNetPlans",
        overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
        suppress_transpose: bool = False,
    ):
        super().__init__(
            dataset_name_or_id,
            gpu_memory_target_in_gb,
            preprocessor_name,
            plans_name,
            overwrite_target_spacing,
            suppress_transpose,
        )

    @classmethod
    def _phase_one_configuration(cls, preset: dict) -> dict:
        return {
            "inherits_from": preset["inherits_from"],
            "patch_size_multiplier": preset["patch_size_multiplier"],
            "architecture": {
                "arch_kwargs": {
                    "features_per_stage": preset["features_per_stage"],
                },
            },
        }

    def _additional_configurations(self) -> dict:
        return {
            name: self._phase_one_configuration(preset)
            for name, preset in self.phase_one_presets.items()
        }


class _DeprecatedPhasePlanner(CondUNetPlanner):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"{type(self).__name__} is deprecated because its presets target CondUNet "
            "features that have been removed. Use CondUNetPlanner or PhaseOnePlanner."
        )


class PhaseTwoPlanner(_DeprecatedPhasePlanner):
    pass


class PhaseThreePlanner(_DeprecatedPhasePlanner):
    pass


class PhaseFourPlanner(_DeprecatedPhasePlanner):
    pass
