import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
)
from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_convtransp,
    maybe_convert_scalar_to_list,
)
from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
    ConvDropoutNormReLU,
    StackedConvBlocks,
)
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

# Type aliases for config flexibility
BoolConfig = Union[bool, List[bool], List[List[bool]]]
IntConfig = Union[int, List[int], Tuple[int, ...]]
KernelConfig = Union[int, List[int], Tuple[int, ...]]


@dataclass
class CCConfig:
    """CondConv (dense mixture-of-experts) addon configuration.

    Attributes:
        enabled: Per-block CC enablement in the encoder.
        num_experts: Number of experts per encoder stage (int or per-stage list).
        reduction: Optional reduction factor for the router MLP hidden width.
        use_std: Include global standard deviation in the routing descriptor.
        expert_dropout: Probability of dropping an expert routing score.
        num_groups: Number of independently routed channel groups.
    """

    enabled: BoolConfig = False
    num_experts: IntConfig = 0
    reduction: Optional[float] = None
    use_std: bool = False
    expert_dropout: float = 0.0
    num_groups: int = 1

    def __post_init__(self):
        if self.reduction is not None:
            self.reduction = _validate_reduction(self.reduction, "cc.reduction")
        self.use_std = _validate_bool(self.use_std, "cc.use_std")
        self.expert_dropout = _validate_probability(
            self.expert_dropout, "cc.expert_dropout"
        )
        self.num_groups = _validate_positive_int(self.num_groups, "cc.num_groups")

    @classmethod
    def from_dict(cls, d: dict) -> "CCConfig":
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StemConfig:
    """Downsampling stem configuration.

    Attributes:
        channels: Number of output channels for the stem. Defaults to
            ``features_per_stage[0]`` when None.
        kernel_size: Convolution kernel size for the stem.
        stride: Convolution stride for the stem (controls initial downsampling).
    """

    channels: Optional[int] = None
    kernel_size: KernelConfig = 3
    stride: KernelConfig = 1

    @classmethod
    def from_dict(cls, d: dict) -> "StemConfig":
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_config(value, config_class):
    """Convert a plain dict to the corresponding config dataclass, or return as-is if already an instance."""
    if value is None:
        return config_class()
    if isinstance(value, dict):
        return config_class.from_dict(value)
    if isinstance(value, config_class):
        return value
    raise TypeError(
        f"Expected {config_class.__name__}, dict, or None, got {type(value).__name__}"
    )


def _validate_reduction(value: float, name: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 1
    ):
        raise ValueError(f"{name} must be a finite number greater than or equal to 1")
    return float(value)


def _validate_bool(value: bool, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _validate_probability(value: float, name: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        or value >= 1
    ):
        raise ValueError(f"{name} must be a finite number in the range [0, 1)")
    return float(value)


def _validate_drop_rate(value: float) -> float:
    return _validate_probability(value, "drop_rate")


def _validate_positive_int(value: int, name: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _same_padding(
    kernel_size: Union[int, List[int], Tuple[int, ...]],
) -> Union[int, List[int]]:
    values = [kernel_size] if isinstance(kernel_size, int) else list(kernel_size)
    if any(
        not isinstance(value, (int, np.integer))
        or isinstance(value, bool)
        or value <= 0
        for value in values
    ):
        raise ValueError("kernel size values must be positive integers")
    if any(value % 2 == 0 for value in values):
        raise ValueError("kernel size values must be odd when using same padding")
    padding = [(int(value) - 1) // 2 for value in values]
    return padding[0] if isinstance(kernel_size, int) else padding


def _normalize_spatial_param(
    conv_op: Type[_ConvNd],
    value: Union[int, Sequence[int]],
    name: str,
    *,
    require_odd: bool = False,
) -> List[int]:
    dim = convert_conv_op_to_dim(conv_op)
    values = maybe_convert_scalar_to_list(conv_op, value)
    if len(values) != dim:
        raise ValueError(f"{name} must contain exactly {dim} values, got {len(values)}")
    if any(
        not isinstance(item, (int, np.integer)) or isinstance(item, bool) or item <= 0
        for item in values
    ):
        raise ValueError(f"{name} values must be positive integers")
    if require_odd and any(item % 2 == 0 for item in values):
        raise ValueError(f"{name} values must be odd when using same padding")
    return [int(item) for item in values]


def _transpose_output_padding(
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
) -> List[int]:
    output_padding = [s - k + 2 * p for k, s, p in zip(kernel_size, stride, padding)]
    for op, s, k in zip(output_padding, stride, kernel_size):
        if op < 0 or op >= s:
            raise ValueError(
                f"stem_kernel_size={kernel_size} and stem_stride={stride} cannot be inverted by "
                "a matching transposed convolution. Use a larger stem stride or an odd stem kernel."
            )
    return output_padding


def _validate_native_resolution_decoder(strides: List[List[int]]) -> None:
    if any(i != 1 for i in strides[0]):
        raise ValueError(
            "The first encoder stage must use stride 1. The segmentation head only inverts the "
            "patchify stem, so a strided first stage would leave predictions below native resolution."
        )


def _kernel_size_for_dim(kernel_size: Union[int, Sequence[int]], dim: int) -> List[int]:
    if isinstance(kernel_size, int):
        return [kernel_size] * dim
    kernel_size = list(kernel_size)
    if len(kernel_size) >= dim:
        return kernel_size[:dim]
    return kernel_size + [1] * (dim - len(kernel_size))


def _normalize_kernel_sizes(
    conv_op: Type[_ConvNd],
    kernel_sizes: Union[None, int, Sequence[int], Sequence[Sequence[int]]],
    n_stages: int,
    default_kernel_size: Sequence[int],
) -> List[List[int]]:
    dim = convert_conv_op_to_dim(conv_op)
    if kernel_sizes is None:
        normalized = [
            _kernel_size_for_dim(default_kernel_size, dim) for _ in range(n_stages)
        ]
        return [
            _normalize_spatial_param(conv_op, value, "kernel_sizes", require_odd=True)
            for value in normalized
        ]
    if isinstance(kernel_sizes, int):
        normalized = [
            maybe_convert_scalar_to_list(conv_op, kernel_sizes) for _ in range(n_stages)
        ]
        return [
            _normalize_spatial_param(conv_op, value, "kernel_sizes", require_odd=True)
            for value in normalized
        ]

    kernel_sizes = list(kernel_sizes)
    if len(kernel_sizes) == 0:
        raise ValueError("kernel_sizes must not be empty")
    if isinstance(kernel_sizes[0], (list, tuple)):
        if len(kernel_sizes) != n_stages:
            raise ValueError(
                f"Expected one kernel size per stage ({n_stages}), got {len(kernel_sizes)}"
            )
        normalized = [_kernel_size_for_dim(i, dim) for i in kernel_sizes]
        return [
            _normalize_spatial_param(conv_op, value, "kernel_sizes", require_odd=True)
            for value in normalized
        ]
    if len(kernel_sizes) == dim:
        normalized = [_kernel_size_for_dim(kernel_sizes, dim) for _ in range(n_stages)]
        return [
            _normalize_spatial_param(conv_op, value, "kernel_sizes", require_odd=True)
            for value in normalized
        ]
    if len(kernel_sizes) == n_stages:
        normalized = [
            maybe_convert_scalar_to_list(conv_op, int(i)) for i in kernel_sizes
        ]
        return [
            _normalize_spatial_param(conv_op, value, "kernel_sizes", require_odd=True)
            for value in normalized
        ]
    raise ValueError(
        f"Cannot interpret kernel_sizes={kernel_sizes}. Provide one {dim}D kernel or one kernel per stage."
    )


def _interpolation_mode(conv_op: Type[_ConvNd]) -> str:
    dim = convert_conv_op_to_dim(conv_op)
    if dim == 3:
        return "trilinear"
    if dim == 2:
        return "bilinear"
    return "linear"


def _conv_output_shape(
    input_size: Sequence[int], kernel_size: Sequence[int], stride: Sequence[int]
) -> List[int]:
    padding = _same_padding(kernel_size)
    return [
        (size + 2 * pad - kernel) // step + 1
        for size, kernel, step, pad in zip(input_size, kernel_size, stride, padding)
    ]


class Router(nn.Module):
    """Global descriptor projected to normalized grouped expert scores."""

    def __init__(
        self,
        input_channels: int,
        num_experts: int,
        num_groups: int = 1,
        reduction: Optional[float] = None,
        use_std: bool = False,
        expert_dropout: float = 0.0,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
    ):
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be greater than 0")
        self.num_experts = _validate_positive_int(num_experts, "num_experts")
        self.num_groups = _validate_positive_int(num_groups, "num_groups")
        self.use_std = _validate_bool(use_std, "use_std")
        self.expert_dropout = _validate_probability(
            expert_dropout, "expert_dropout"
        )
        if reduction is not None:
            reduction = _validate_reduction(reduction, "reduction")

        self.input_channels = input_channels
        self.descriptor_channels = input_channels * (2 if self.use_std else 1)
        self.output_channels = self.num_groups * self.num_experts
        self.reduction = reduction
        if reduction is None:
            self.hidden_channels = None
            self.input_projection = None
            self.nonlin = None
            self.output_projection = nn.Linear(
                self.descriptor_channels, self.output_channels
            )
        else:
            self.hidden_channels = max(1, int(self.descriptor_channels / reduction))
            self.input_projection = nn.Linear(
                self.descriptor_channels, self.hidden_channels
            )
            self.nonlin = (
                nonlin(**({} if nonlin_kwargs is None else nonlin_kwargs))
                if nonlin is not None
                else nn.Identity()
            )
            self.output_projection = nn.Linear(
                self.hidden_channels, self.output_channels
            )
        self.register_buffer(
            "_active_expert_dropout",
            torch.tensor(self.expert_dropout, dtype=torch.float32),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.output_projection.bias)
        nn.init.zeros_(self.output_projection.weight)

    @property
    def active_expert_dropout(self) -> float:
        return float(self._active_expert_dropout.item())

    def set_expert_dropout(self, value: float) -> None:
        value = _validate_probability(value, "expert_dropout")
        self._active_expert_dropout.fill_(value)

    def _descriptor(self, x: torch.Tensor) -> torch.Tensor:
        spatial_dims = tuple(range(2, x.ndim))
        if self.use_std:
            std, mean = torch.std_mean(x, dim=spatial_dims, correction=0)
            return torch.cat((std, mean), dim=1)
        return torch.mean(x, dim=spatial_dims)

    def _drop_experts(self, scores: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return scores
        keep_mask = torch.rand_like(scores) >= self._active_expert_dropout
        invalid = ~keep_mask.any(dim=-1)
        while invalid.any():
            keep_mask[invalid] = (
                torch.rand_like(scores[invalid]) >= self._active_expert_dropout
            )
            invalid = ~keep_mask.any(dim=-1)
        return scores * keep_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptor = self._descriptor(x)
        if self.input_projection is None:
            logits = self.output_projection(descriptor)
        else:
            logits = self.output_projection(
                self.nonlin(self.input_projection(descriptor))
            )
        scores = torch.sigmoid(logits).reshape(
            x.shape[0], self.num_groups, self.num_experts
        )
        scores = self._drop_experts(scores)
        return scores / scores.sum(dim=-1, keepdim=True)


class CondPWConv(nn.Module):
    """Pointwise convolution with independently routed channel groups."""

    def __init__(
        self,
        conv: _ConvNd,
        num_experts: int,
        num_groups: int = 1,
        group_axis: str = "output",
    ):
        super().__init__()
        self.num_experts = _validate_positive_int(num_experts, "num_experts")
        self.num_groups = _validate_positive_int(num_groups, "num_groups")
        if group_axis not in ("input", "output"):
            raise ValueError("group_axis must be 'input' or 'output'")
        if conv.groups != 1 or any(k != 1 for k in conv.kernel_size):
            raise ValueError(
                "CondPWConv is only compatible with dense pointwise convolutions."
            )
        if conv.bias is not None:
            raise ValueError(
                "CondPWConv only supports bias-free pointwise convolutions."
            )

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.spatial_dims = len(conv.kernel_size)
        self.group_axis = group_axis
        grouped_channels = (
            self.out_channels if group_axis == "output" else self.in_channels
        )
        if grouped_channels % self.num_groups:
            raise ValueError(
                f"{group_axis}_channels ({grouped_channels}) must be divisible by "
                f"num_groups ({self.num_groups})"
            )
        self.expert_weights = nn.Parameter(
            conv.weight.new_empty(
                self.num_experts, self.out_channels, self.in_channels
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        for expert_weight in self.expert_weights:
            nn.init.kaiming_uniform_(expert_weight, a=math.sqrt(5))

    def _blend_experts(self, scores: torch.Tensor) -> torch.Tensor:
        if self.group_axis == "output":
            weights = self.expert_weights.reshape(
                self.num_experts,
                self.num_groups,
                self.out_channels // self.num_groups,
                self.in_channels,
            )
            return torch.einsum("bgk,kgoi->bgoi", scores, weights).reshape(
                scores.shape[0], self.out_channels, self.in_channels
            )
        weights = self.expert_weights.reshape(
            self.num_experts,
            self.out_channels,
            self.num_groups,
            self.in_channels // self.num_groups,
        )
        return torch.einsum("bgk,kogj->bogj", scores, weights).reshape(
            scores.shape[0], self.out_channels, self.in_channels
        )

    def _validate_scores(self, x: torch.Tensor, scores: torch.Tensor) -> None:
        if scores.shape[0] != x.shape[0]:
            raise ValueError(
                f"router score batch size ({scores.shape[0]}) does not match input batch size ({x.shape[0]})"
            )
        expected_shape = (x.shape[0], self.num_groups, self.num_experts)
        if tuple(scores.shape) != expected_shape:
            raise ValueError(
                f"router scores must have shape {expected_shape}, got "
                f"{tuple(scores.shape)}"
            )

    def forward(self, x: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self._validate_scores(x, scores)
        weight = self._blend_experts(scores)
        flat_input = x.flatten(2)
        return torch.bmm(weight, flat_input).reshape(
            x.shape[0], self.out_channels, *x.shape[2:]
        )


def _expand_expansion_ratios(
    value: Union[float, Sequence[float]], n_stages: int, name: str
) -> List[float]:
    values = [value] * n_stages if isinstance(value, Real) else list(value)
    if len(values) != n_stages:
        raise ValueError(
            f"{name} must contain exactly {n_stages} values, got {len(values)}"
        )
    if any(
        not isinstance(item, Real) or isinstance(item, bool) or item <= 0
        for item in values
    ):
        raise ValueError(f"{name} values must be positive numbers")
    return [float(item) for item in values]


def _expand_int_param(
    value: Union[int, Sequence[int]], n_stages: int, name: str, *, min_value: int = 0
) -> List[int]:
    """Expand a scalar-or-sequence of ints to a per-stage list, validating >= min_value."""
    values = [value] * n_stages if isinstance(value, (int, np.integer)) else list(value)
    if len(values) != n_stages:
        raise ValueError(
            f"{name} must contain exactly {n_stages} values, got {len(values)}"
        )
    if any(
        not isinstance(item, (int, np.integer))
        or isinstance(item, bool)
        or item < min_value
        for item in values
    ):
        raise ValueError(f"{name} values must be integers >= {min_value}")
    return [int(v) for v in values]


def _expand_block_config(
    config: Union[bool, Sequence[Union[bool, Sequence[bool]]]],
    n_stages: int,
    n_blocks_per_stage: List[int],
    name: str,
) -> List[List[bool]]:
    """Helper to convert varying granularity configurations into a 2D list representing block-level usage.

    The config can be:
    - A single boolean (applied to all blocks across all stages).
    - A 1D sequence of booleans (one boolean per stage, applied to all blocks in that stage).
    - A nested sequence of booleans (a sequence of sequences matching `n_blocks_per_stage`).
    """
    if isinstance(config, (bool, np.bool_)):
        return [[bool(config)] * n for n in n_blocks_per_stage]

    if not isinstance(config, (list, tuple, np.ndarray)):
        raise ValueError(
            f"{name} must be a boolean or a sequence of booleans / sequences of booleans, "
            f"got {type(config)} instead."
        )

    config = list(config)
    if len(config) != n_stages:
        raise ValueError(
            f"Expected {name} to contain exactly {n_stages} values (one per stage), "
            f"but got a length of {len(config)}."
        )

    result = []
    for stage_idx, stage_config in enumerate(config):
        n_blocks = n_blocks_per_stage[stage_idx]
        if isinstance(stage_config, (bool, np.bool_)):
            result.append([bool(stage_config)] * n_blocks)
        elif isinstance(stage_config, (list, tuple, np.ndarray)):
            stage_config = list(stage_config)
            if len(stage_config) != n_blocks:
                raise ValueError(
                    f"Expected {name}[{stage_idx}] to contain exactly {n_blocks} values (one per block), "
                    f"but got a length of {len(stage_config)}."
                )
            if not all(isinstance(val, (bool, np.bool_)) for val in stage_config):
                raise ValueError(
                    f"All values inside block configuration {name}[{stage_idx}] must be booleans. "
                    f"Got {stage_config}."
                )
            result.append([bool(val) for val in stage_config])
        else:
            raise ValueError(
                f"Unexpected value type '{type(stage_config)}' inside {name} for stage {stage_idx}. "
                "Expected a boolean or a sequence of booleans."
            )
    return result


@dataclass(frozen=True)
class _StageSettings:
    n_blocks: int
    expansion_ratio: float
    num_experts: int
    cc_blocks: Tuple[bool, ...]


def _normalize_stage_settings(
    n_stages: int,
    n_blocks_per_stage: Union[int, Sequence[int]],
    expansion_ratio: Union[float, Sequence[float]],
    num_experts: Union[int, Sequence[int]],
    cc: BoolConfig,
    context: str,
) -> List[_StageSettings]:
    block_counts = _expand_int_param(
        n_blocks_per_stage, n_stages, f"{context} n_blocks_per_stage", min_value=1
    )
    expansion_ratios = _expand_expansion_ratios(
        expansion_ratio, n_stages, f"{context} expansion_ratio"
    )
    expert_counts = _expand_int_param(
        num_experts, n_stages, f"{context} num_experts", min_value=0
    )
    cc_blocks = _expand_block_config(cc, n_stages, block_counts, f"{context} cc")

    settings = []
    for stage_idx in range(n_stages):
        if any(cc_blocks[stage_idx]) and expert_counts[stage_idx] == 0:
            raise ValueError(
                f"CondConv is enabled in {context} stage {stage_idx} "
                f"(block-level config: {cc_blocks[stage_idx]}), but num_experts is 0"
            )
        settings.append(
            _StageSettings(
                n_blocks=block_counts[stage_idx],
                expansion_ratio=expansion_ratios[stage_idx],
                num_experts=expert_counts[stage_idx],
                cc_blocks=tuple(cc_blocks[stage_idx]),
            )
        )
    return settings


def _schedule_encoder_drop_rates(
    stage_settings: Sequence[_StageSettings], drop_rate: float
) -> List[List[float]]:
    drop_rate = _validate_drop_rate(drop_rate)
    scheduled = [[0.0] * settings.n_blocks for settings in stage_settings]
    eligible = [(0, 0)] + [
        (stage_idx, block_idx)
        for stage_idx, settings in enumerate(stage_settings)
        for block_idx in range(1, settings.n_blocks)
    ]
    if len(eligible) == 1:
        return scheduled
    denominator = len(eligible) - 1
    for position, (stage_idx, block_idx) in enumerate(eligible):
        scheduled[stage_idx][block_idx] = drop_rate * position / denominator
    return scheduled


def _forward_routed_conv_block(
    conv_block: ConvDropoutNormReLU, x: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    for module in conv_block.all_modules:
        x = module(x, scores) if isinstance(module, CondPWConv) else module(x)
    return x


def _expertify_pointwise(
    conv_block: ConvDropoutNormReLU,
    num_experts: int,
    num_groups: int,
    group_axis: str,
) -> None:
    conv = conv_block.conv
    if conv.groups != 1 or any(kernel_size != 1 for kernel_size in conv.kernel_size):
        raise ValueError("CondMobileNet only expertifies dense pointwise convolutions")
    conditional_conv = CondPWConv(
        conv, num_experts, num_groups=num_groups, group_axis=group_axis
    )
    conv_block.conv = conditional_conv
    conv_block.all_modules[0] = conditional_conv


class DepthwiseConvBlock(nn.Module):
    """Depthwise block: DW Conv -> Norm -> Activation -> Dropout."""

    def __init__(
        self,
        conv_op: Type[_ConvNd],
        channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]],
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
    ):
        super().__init__()
        if (
            not isinstance(channels, (int, np.integer))
            or isinstance(channels, bool)
            or channels <= 0
        ):
            raise ValueError("channels must be a positive integer")
        kernel_size = _normalize_spatial_param(
            conv_op, kernel_size, "kernel_size", require_odd=True
        )
        stride = _normalize_spatial_param(conv_op, stride, "stride")
        norm_op_kwargs = {} if norm_op_kwargs is None else norm_op_kwargs
        nonlin_kwargs = {} if nonlin_kwargs is None else nonlin_kwargs
        dropout_op_kwargs = {} if dropout_op_kwargs is None else dropout_op_kwargs
        self.conv = conv_op(
            channels,
            channels,
            kernel_size,
            stride,
            padding=_same_padding(kernel_size),
            groups=channels,
            bias=False,
        )
        self.norm = (
            norm_op(channels, **norm_op_kwargs)
            if norm_op is not None
            else nn.Identity()
        )
        self.nonlin = nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()
        self.dropout = (
            dropout_op(**dropout_op_kwargs)
            if dropout_op is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.nonlin(self.norm(self.conv(x))))


class InvertedBottleneckBlock(nn.Module):
    """Inverted bottleneck with optional grouped pointwise expert routing."""

    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]],
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        expansion_ratio: float = 3.0,
        num_experts: int = 0,
        cc: bool = False,
        reduction: Optional[float] = None,
        use_std: bool = False,
        expert_dropout: float = 0.0,
        num_groups: int = 1,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = _normalize_spatial_param(conv_op, stride, "stride")
        kernel_size = _normalize_spatial_param(
            conv_op, kernel_size, "kernel_size", require_odd=True
        )
        norm_op_kwargs = {} if norm_op_kwargs is None else norm_op_kwargs
        nonlin_kwargs = {} if nonlin_kwargs is None else nonlin_kwargs

        self.expanded_channels = int(round(expansion_ratio * input_channels))
        if self.expanded_channels <= 0:
            raise ValueError(
                f"expansion_ratio must produce at least one channel, got {expansion_ratio}"
            )

        # PW expansion: PW -> Norm/Act
        self.expand = ConvDropoutNormReLU(
            conv_op,
            input_channels,
            self.expanded_channels,
            1,
            1,
            False,
            norm_op,
            norm_op_kwargs,
            None,
            None,
            nonlin,
            nonlin_kwargs,
        )

        # Depthwise DW: DW -> Norm/Act
        self.depthwise = DepthwiseConvBlock(
            conv_op,
            self.expanded_channels,
            kernel_size,
            self.stride,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
            dropout_op,
            dropout_op_kwargs,
        )

        # PW projection: PW -> Norm
        self.project = ConvDropoutNormReLU(
            conv_op,
            self.expanded_channels,
            output_channels,
            1,
            1,
            False,
            norm_op,
            norm_op_kwargs,
            None,
            None,
            None,
            None,
        )
        self.add_identity = input_channels == output_channels and all(
            i == 1 for i in self.stride
        )
        self.drop_rate = _validate_drop_rate(drop_rate)

        self.num_experts = num_experts if cc else 0
        if cc:
            if num_experts <= 0:
                raise ValueError(
                    f"CondConv is enabled (cc=True) but num_experts is {num_experts}. "
                    "num_experts must be greater than 0 to configure a dynamic mixture of experts."
                )
            self.router = Router(
                input_channels,
                num_experts,
                num_groups=num_groups,
                reduction=reduction,
                use_std=use_std,
                expert_dropout=expert_dropout,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
            _expertify_pointwise(
                self.expand, num_experts, num_groups, group_axis="output"
            )
            _expertify_pointwise(
                self.project, num_experts, num_groups, group_axis="input"
            )
        else:
            self.router = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.router is not None:
            scores = self.router(x)
            x = _forward_routed_conv_block(self.expand, x, scores)
        else:
            x = self.expand(x)
        x = self.depthwise(x)
        if self.router is not None:
            x = self.project.conv(x, scores)
        else:
            x = self.project.conv(x)

        if hasattr(self.project, "norm"):
            x = self.project.norm(x)
        if self.add_identity:
            if self.training and self.drop_rate > 0:
                keep_probability = 1 - self.drop_rate
                mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                keep_mask = torch.empty(
                    mask_shape, dtype=x.dtype, device=x.device
                ).bernoulli_(keep_probability)
                x = x * keep_mask / keep_probability
            x = x + residual
        return x

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), (
            "just give the image size without color/feature channels or batch channel. "
            "Do not give input_size=(b, c, x, y(, z)). Give input_size=(x, y(, z))!"
        )
        size_after_stride = _conv_output_shape(
            input_size, self.depthwise.conv.kernel_size, self.stride
        )
        output = np.prod([self.expanded_channels, *input_size], dtype=np.int64)
        output += np.prod([self.expanded_channels, *size_after_stride], dtype=np.int64)
        output += np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        return output


class StackedCondInvertedBottleneckBlocks(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        initial_stride: Union[int, List[int], Tuple[int, ...]],
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        expansion_ratio: float = 3.0,
        num_experts: int = 0,
        cc_config: List[bool] = None,
        reduction: Optional[float] = None,
        use_std: bool = False,
        expert_dropout: float = 0.0,
        num_groups: int = 1,
        drop_rates: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        if (
            not isinstance(n_blocks, (int, np.integer))
            or isinstance(n_blocks, bool)
            or n_blocks <= 0
        ):
            raise ValueError("n_blocks must be greater than 0")
        self.initial_stride = _normalize_spatial_param(
            conv_op, initial_stride, "initial_stride"
        )
        self.output_channels = output_channels
        if cc_config is None:
            cc_config = [False] * n_blocks
        if len(cc_config) != n_blocks:
            raise ValueError(
                f"cc_config length ({len(cc_config)}) must match n_blocks ({n_blocks})"
            )
        if drop_rates is None:
            drop_rates = [0.0] * n_blocks
        if len(drop_rates) != n_blocks:
            raise ValueError(
                f"drop_rates length ({len(drop_rates)}) must match n_blocks ({n_blocks})"
            )

        def make_block(
            block_idx: int,
            in_channels: int,
            stride,
            block_cc: bool,
        ):
            return InvertedBottleneckBlock(
                conv_op=conv_op,
                input_channels=in_channels,
                output_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                expansion_ratio=expansion_ratio,
                num_experts=num_experts,
                cc=block_cc,
                reduction=reduction,
                use_std=use_std,
                expert_dropout=expert_dropout,
                num_groups=num_groups,
                drop_rate=drop_rates[block_idx],
            )

        self.blocks = nn.ModuleList(
            (
                make_block(
                    0,
                    input_channels,
                    initial_stride,
                    cc_config[0],
                ),
                *[
                    make_block(i, output_channels, 1, cc_config[i])
                    for i in range(1, n_blocks)
                ],
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def compute_conv_feature_map_size(self, input_size):
        output = self.blocks[0].compute_conv_feature_map_size(input_size)
        size_after_stride = _conv_output_shape(
            input_size, self.blocks[0].depthwise.conv.kernel_size, self.initial_stride
        )
        for block in self.blocks[1:]:
            output += block.compute_conv_feature_map_size(size_after_stride)
        return output


class CondUNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        expansion_ratio: Union[float, Sequence[float]] = 3.0,
        return_skips: bool = True,
        stem_channels: int = None,
        stem_kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        stem_stride: Union[int, List[int], Tuple[int, ...]] = 1,
        num_experts: Union[int, Sequence[int]] = 0,
        cc: BoolConfig = False,
        cc_reduction: Optional[float] = None,
        cc_use_std: bool = False,
        cc_expert_dropout: float = 0.0,
        cc_num_groups: int = 1,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        if (
            not isinstance(n_stages, (int, np.integer))
            or isinstance(n_stages, bool)
            or n_stages <= 0
        ):
            raise ValueError("n_stages must be a positive integer")
        if (
            not isinstance(input_channels, (int, np.integer))
            or isinstance(input_channels, bool)
            or input_channels <= 0
        ):
            raise ValueError("input_channels must be a positive integer")
        if isinstance(features_per_stage, int):
            raise TypeError(
                f"features_per_stage must be explicitly provided as a sequence of integers, "
                f"not a single integer: {features_per_stage}"
            )
        features_per_stage = list(features_per_stage)
        if len(features_per_stage) != n_stages:
            raise ValueError(
                f"features_per_stage must contain exactly {n_stages} values"
            )
        if any(
            not isinstance(channels, (int, np.integer))
            or isinstance(channels, bool)
            or channels <= 0
            for channels in features_per_stage
        ):
            raise ValueError("features_per_stage values must be positive integers")
        features_per_stage = [int(channels) for channels in features_per_stage]

        raw_strides = (
            [strides] * n_stages if isinstance(strides, int) else list(strides)
        )
        if len(raw_strides) != n_stages:
            raise ValueError(f"strides must contain exactly {n_stages} values")
        strides = [
            _normalize_spatial_param(conv_op, stride, f"strides[{stage_idx}]")
            for stage_idx, stride in enumerate(raw_strides)
        ]
        kernel_sizes = _normalize_kernel_sizes(
            conv_op, kernel_sizes, n_stages, [3] * convert_conv_op_to_dim(conv_op)
        )
        stage_settings = _normalize_stage_settings(
            n_stages,
            n_blocks_per_stage,
            expansion_ratio,
            num_experts,
            cc,
            "encoder",
        )
        stage_drop_rates = _schedule_encoder_drop_rates(stage_settings, drop_rate)
        self.num_experts = [settings.num_experts for settings in stage_settings]
        self.expansion_ratios = [
            settings.expansion_ratio for settings in stage_settings
        ]
        self.cc_config = [list(settings.cc_blocks) for settings in stage_settings]

        stem_channels = (
            features_per_stage[0] if stem_channels is None else stem_channels
        )
        if (
            not isinstance(stem_channels, (int, np.integer))
            or isinstance(stem_channels, bool)
            or stem_channels <= 0
        ):
            raise ValueError("stem.channels must be a positive integer or None")
        stem_channels = int(stem_channels)
        self.stem_kernel_size = _normalize_spatial_param(
            conv_op, stem_kernel_size, "stem.kernel_size", require_odd=True
        )
        self.stem_stride = _normalize_spatial_param(conv_op, stem_stride, "stem.stride")

        # Stem applies no non-linearity (only Conv + Norm).
        self.stem = StackedConvBlocks(
            1,
            conv_op,
            input_channels,
            stem_channels,
            self.stem_kernel_size,
            self.stem_stride,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            None,
            None,
        )

        stages = []
        stage_input_channels = stem_channels
        for stage_idx, settings in enumerate(stage_settings):
            stages.append(
                StackedCondInvertedBottleneckBlocks(
                    n_blocks=settings.n_blocks,
                    conv_op=conv_op,
                    input_channels=stage_input_channels,
                    output_channels=features_per_stage[stage_idx],
                    kernel_size=kernel_sizes[stage_idx],
                    initial_stride=strides[stage_idx],
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    expansion_ratio=settings.expansion_ratio,
                    num_experts=settings.num_experts,
                    cc_config=list(settings.cc_blocks),
                    reduction=cc_reduction,
                    use_std=cc_use_std,
                    expert_dropout=cc_expert_dropout,
                    num_groups=cc_num_groups,
                    drop_rates=stage_drop_rates[stage_idx],
                )
            )
            stage_input_channels = features_per_stage[stage_idx]

        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = strides
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips if self.return_skips else skips[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = self.stem.compute_conv_feature_map_size(input_size)
        input_size = _conv_output_shape(
            input_size, self.stem_kernel_size, self.stem_stride
        )
        for stage_idx, stage in enumerate(self.stages):
            output += stage.compute_conv_feature_map_size(input_size)
            input_size = _conv_output_shape(
                input_size, self.kernel_sizes[stage_idx], self.strides[stage_idx]
            )
        return output

    def compute_bottleneck_shape(self, input_size: Sequence[int]) -> Tuple[int, ...]:
        spatial_shape = _conv_output_shape(
            input_size, self.stem_kernel_size, self.stem_stride
        )
        for kernel_size, stride in zip(self.kernel_sizes, self.strides):
            spatial_shape = _conv_output_shape(spatial_shape, kernel_size, stride)
        return tuple(int(size) for size in spatial_shape)


class CondUNetDecoder(nn.Module):
    def __init__(
        self,
        encoder: CondUNetEncoder,
        num_classes: int,
        n_blocks_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        expansion_ratio: Union[float, Sequence[float]] = 3.0,
    ):
        super().__init__()
        if (
            not isinstance(num_classes, (int, np.integer))
            or isinstance(num_classes, bool)
            or num_classes <= 0
        ):
            raise ValueError("num_classes must be a positive integer")
        num_classes = int(num_classes)
        if deep_supervision:
            raise ValueError(
                "CondUNet does not support deep supervision; set deep_supervision=False"
            )
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.interp_mode = _interpolation_mode(encoder.conv_op)
        n_stages_encoder = len(encoder.output_channels)
        _validate_native_resolution_decoder(encoder.strides)
        stage_settings = _normalize_stage_settings(
            n_stages_encoder - 1,
            n_blocks_per_stage,
            expansion_ratio,
            0,
            False,
            "decoder",
        )
        self.expansion_ratios = [
            settings.expansion_ratio for settings in stage_settings
        ]

        stages = []
        for s in range(1, n_stages_encoder):
            settings = stage_settings[s - 1]
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stage_input_channels = input_features_below + input_features_skip
            target_stage_idx = n_stages_encoder - s - 1
            stages.append(
                StackedCondInvertedBottleneckBlocks(
                    n_blocks=settings.n_blocks,
                    conv_op=encoder.conv_op,
                    input_channels=stage_input_channels,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[target_stage_idx],
                    initial_stride=1,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    dropout_op=encoder.dropout_op,
                    dropout_op_kwargs=encoder.dropout_op_kwargs,
                    expansion_ratio=settings.expansion_ratio,
                )
            )

        self.stages = nn.ModuleList(stages)
        head_op = get_matching_convtransp(conv_op=encoder.conv_op)
        head_padding = _same_padding(encoder.stem_kernel_size)
        head_output_padding = _transpose_output_padding(
            encoder.stem_kernel_size, encoder.stem_stride, head_padding
        )
        self.seg_norm = (
            encoder.norm_op(
                encoder.output_channels[0], **(encoder.norm_op_kwargs or {})
            )
            if encoder.norm_op is not None
            else nn.Identity()
        )
        self.seg_layer = head_op(
            encoder.output_channels[0],
            num_classes,
            encoder.stem_kernel_size,
            encoder.stem_stride,
            padding=head_padding,
            output_padding=head_output_padding,
            bias=True,
        )

    def forward(self, skips):
        if self.deep_supervision:
            raise RuntimeError("CondUNet does not support deep supervision")
        x = skips[-1]
        for stage_idx, stage in enumerate(self.stages):
            skip = skips[-(stage_idx + 2)]
            x = F.interpolate(
                x, size=skip.shape[2:], mode=self.interp_mode, align_corners=False
            )
            x = stage(torch.cat((x, skip), dim=1))
        seg_output = self.seg_layer(self.seg_norm(x))
        return seg_output

    def compute_conv_feature_map_size(self, input_size):
        native_input_size = input_size
        input_size = _conv_output_shape(
            input_size, self.encoder.stem_kernel_size, self.encoder.stem_stride
        )
        skip_sizes = []
        for kernel_size, stride in zip(
            self.encoder.kernel_sizes[:-1], self.encoder.strides[:-1]
        ):
            input_size = _conv_output_shape(input_size, kernel_size, stride)
            skip_sizes.append(input_size)

        output = np.int64(0)
        for stage_idx, stage in enumerate(self.stages):
            skip_size = skip_sizes[-(stage_idx + 1)]
            output += stage.compute_conv_feature_map_size(skip_size)
        output += np.prod([self.num_classes, *native_input_size], dtype=np.int64)
        return output


class CondUNet(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        encoder_n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        decoder_n_blocks_per_stage: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        encoder_expansion_ratio: Union[float, Sequence[float]] = 3.0,
        decoder_expansion_ratio: Union[float, Sequence[float]] = 3.0,
        stem: Union[StemConfig, dict, None] = None,
        cc: Union[CCConfig, dict, None] = None,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.key_to_encoder = "encoder.stages"
        self.key_to_stem = "encoder.stem"
        self.keys_to_in_proj = (
            "encoder.stem.convs.0.conv",
            "encoder.stem.convs.0.all_modules.0",
        )

        if isinstance(features_per_stage, int):
            raise TypeError(
                f"features_per_stage must be explicitly provided as a sequence of integers, "
                f"not a single integer: {features_per_stage}"
            )

        # Normalize config dataclasses (supports dict from JSON plans)
        stem = _normalize_config(stem, StemConfig)
        cc = _normalize_config(cc, CCConfig)

        self.encoder = CondUNetEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=encoder_n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            expansion_ratio=encoder_expansion_ratio,
            return_skips=True,
            stem_channels=stem.channels,
            stem_kernel_size=stem.kernel_size,
            stem_stride=stem.stride,
            num_experts=cc.num_experts,
            cc=cc.enabled,
            cc_reduction=cc.reduction,
            cc_use_std=cc.use_std,
            cc_expert_dropout=cc.expert_dropout,
            cc_num_groups=cc.num_groups,
            drop_rate=drop_rate,
        )
        self.decoder = CondUNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_blocks_per_stage=decoder_n_blocks_per_stage,
            deep_supervision=deep_supervision,
            expansion_ratio=decoder_expansion_ratio,
        )

    def forward(self, x: torch.Tensor):
        input_spatial_shape = x.shape[2:]
        skips = self.encoder(x)
        output = self.decoder(skips)
        if output.shape[2:] != input_spatial_shape:
            raise ValueError(
                f"CondUNet output spatial shape {tuple(output.shape[2:])} does not match input "
                f"shape {tuple(input_spatial_shape)}. Use input/patch sizes compatible with the configured strides."
            )
        return output

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or batch channel. "
            "Do not give input_size=(b, c, x, y(, z)). Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(
            input_size
        ) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        if isinstance(module, CondPWConv):
            for expert_weight in module.expert_weights:
                nn.init.kaiming_normal_(expert_weight, a=1e-2)
        InitWeights_He(1e-2)(module)
        # Zero-init the projection norm for residual blocks so that the residual
        # path starts as identity. This is safe because CondPWConv is not a
        # subclass of _ConvNd, so InitWeights_He above does not touch it.
        if (
            isinstance(module, InvertedBottleneckBlock)
            and module.add_identity
            and hasattr(module.project, "norm")
        ):
            if module.project.norm.weight is not None:
                nn.init.constant_(module.project.norm.weight, 0)
            if module.project.norm.bias is not None:
                nn.init.constant_(module.project.norm.bias, 0)
