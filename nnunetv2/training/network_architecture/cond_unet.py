import math
from numbers import Real
from typing import List, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
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

from dataclasses import dataclass, field, asdict
from typing import Optional

# Type aliases for config flexibility
BoolConfig = Union[bool, List[bool], List[List[bool]]]
IntConfig = Union[int, List[int], Tuple[int, ...]]
KernelConfig = Union[int, List[int], Tuple[int, ...]]


@dataclass
class SEConfig:
    """Squeeze-and-Excitation addon configuration.

    Attributes:
        reduction: MLP bottleneck ratio for the SE block.
        encoder: Per-block SE enablement in the encoder. A single bool applies
            globally; a list of bools applies per-stage; a nested list applies
            per-block within each stage.
        decoder: Same as ``encoder`` but for the decoder.
        tile_size: Spatial tile size for tiled SE. None means global (standard) SE.
    """
    reduction: float = 0.125
    encoder: BoolConfig = False
    decoder: BoolConfig = False
    tile_size: Union[None, List[int], Tuple[int, ...]] = None

    @classmethod
    def from_dict(cls, d: dict) -> "SEConfig":
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CCConfig:
    """CondConv (dense mixture-of-experts) addon configuration.

    Attributes:
        reduction: MLP bottleneck ratio for the expert router.
        encoder: Per-block CC enablement in the encoder (same format as SEConfig).
        decoder: Per-block CC enablement in the decoder.
        encoder_num_experts: Number of experts per encoder stage (int or per-stage list).
        decoder_num_experts: Number of experts per decoder stage.
        encoder_num_groups: Channel-wise routing granularity per encoder stage.
        decoder_num_groups: Channel-wise routing granularity per decoder stage.
        tile_size: Spatial tile size for tiled routing. None means global routing.
    """
    reduction: float = 0.125
    encoder: BoolConfig = False
    decoder: BoolConfig = False
    encoder_num_experts: IntConfig = 0
    decoder_num_experts: IntConfig = 0
    encoder_num_groups: IntConfig = 1
    decoder_num_groups: IntConfig = 1
    tile_size: Union[None, List[int], Tuple[int, ...]] = None

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


def _same_padding(kernel_size: Union[int, List[int], Tuple[int, ...]]) -> Union[int, List[int]]:
    if isinstance(kernel_size, int):
        return (kernel_size - 1) // 2
    return [(i - 1) // 2 for i in kernel_size]


def _convert_conv_op_to_transposed(conv_op: Type[_ConvNd]) -> Type[nn.Module]:
    dim = convert_conv_op_to_dim(conv_op)
    if dim == 3:
        return nn.ConvTranspose3d
    if dim == 2:
        return nn.ConvTranspose2d
    if dim == 1:
        return nn.ConvTranspose1d
    raise ValueError(f"Unsupported convolution dimensionality: {dim}")


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
        return [_kernel_size_for_dim(default_kernel_size, dim) for _ in range(n_stages)]
    if isinstance(kernel_sizes, int):
        return [maybe_convert_scalar_to_list(conv_op, kernel_sizes) for _ in range(n_stages)]

    kernel_sizes = list(kernel_sizes)
    if len(kernel_sizes) == 0:
        raise ValueError("kernel_sizes must not be empty")
    if isinstance(kernel_sizes[0], (list, tuple)):
        if len(kernel_sizes) != n_stages:
            raise ValueError(f"Expected one kernel size per stage ({n_stages}), got {len(kernel_sizes)}")
        return [_kernel_size_for_dim(i, dim) for i in kernel_sizes]
    if len(kernel_sizes) == dim:
        return [_kernel_size_for_dim(kernel_sizes, dim) for _ in range(n_stages)]
    if len(kernel_sizes) == n_stages:
        return [maybe_convert_scalar_to_list(conv_op, int(i)) for i in kernel_sizes]
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


class SqueezeAndExcitationBlock(nn.Module):
    """Squeeze-and-excitation with independently recalibrated spatial tiles."""

    def __init__(
        self,
        channels: int,
        reduction: float,
        nonlin: Union[None, Type[nn.Module]],
        nonlin_kwargs: dict = None,
        tile_size: Union[None, Sequence[int]] = None,
    ):
        super().__init__()
        if reduction <= 0:
            raise ValueError("reduction must be greater than 0")
        if tile_size is not None and any(tile <= 0 for tile in tile_size):
            raise ValueError("tile_size values must be greater than 0")
        self.tile_size = None if tile_size is None else tuple(tile_size)
        hidden_channels = max(1, round(channels * reduction))
        nonlin_kwargs = {} if nonlin_kwargs is None else nonlin_kwargs
        activation = nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            activation,
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_dims = x.ndim - 2
        if self.tile_size is None:
            scale = self.mlp(x.mean(dim=tuple(range(2, x.ndim))))
            return x * scale.reshape(x.shape[0], x.shape[1], *([1] * spatial_dims))
        if len(self.tile_size) != spatial_dims:
            raise ValueError(
                f"tile_size has {len(self.tile_size)} dimensions, expected {spatial_dims}"
            )

        output_grid = tuple(
            max(1, spatial_size // tile_size)
            for spatial_size, tile_size in zip(x.shape[2:], self.tile_size)
        )
        adaptive_avg_pool = (F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d)[
            spatial_dims - 1
        ]
        scale = self.mlp(adaptive_avg_pool(x, output_grid).movedim(1, -1)).movedim(-1, 1)
        scale = F.interpolate(scale, size=x.shape[2:], mode="nearest")
        return x * scale


class Router(nn.Module):
    """Mean-only router that produces one expert score vector per spatial tile."""

    def __init__(
        self,
        input_channels: int,
        num_experts: int,
        router_reduction: float,
        nonlin: Union[None, Type[nn.Module]],
        nonlin_kwargs: dict = None,
        tile_size: Union[None, Sequence[int]] = None,
    ):
        super().__init__()
        if router_reduction <= 0:
            raise ValueError("router_reduction must be greater than 0")
        if tile_size is not None and any(tile <= 0 for tile in tile_size):
            raise ValueError("tile_size values must be greater than 0")
        self.tile_size = None if tile_size is None else tuple(tile_size)
        hidden_channels = max(1, round(input_channels * router_reduction))
        nonlin_kwargs = {} if nonlin_kwargs is None else nonlin_kwargs
        activation = nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            activation,
            nn.Linear(hidden_channels, num_experts),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tile_size is None:
            return self.mlp(x.mean(dim=tuple(range(2, x.ndim))))
        if len(self.tile_size) != x.ndim - 2:
            raise ValueError(
                f"tile_size has {len(self.tile_size)} dimensions, expected {x.ndim - 2}"
            )

        output_grid = tuple(
            max(1, spatial_size // tile_size)
            for spatial_size, tile_size in zip(x.shape[2:], self.tile_size)
        )
        adaptive_avg_pool = (F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d)[
            x.ndim - 3
        ]
        return self.mlp(adaptive_avg_pool(x, output_grid).movedim(1, -1))


class CondPWConv(nn.Module):
    """Pointwise mixture-of-experts convolution evaluated via batched matrix multiplication (BMM)."""

    def __init__(
        self,
        conv: _ConvNd,
        num_experts: int,
        router_reduction: float,
        nonlin: Union[None, Type[nn.Module]],
        nonlin_kwargs: dict = None,
        use_internal_router: bool = True,
        num_groups: int = 1,
        group_on_out: bool = True,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be greater than 0")
        if router_reduction <= 0:
            raise ValueError("router_reduction must be greater than 0")
        if num_groups <= 0:
            raise ValueError("num_groups must be greater than 0")

        if conv.groups != 1 or any(k != 1 for k in conv.kernel_size):
            raise ValueError("CondPWConv is only compatible with dense pointwise convolutions.")

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.spatial_dims = len(conv.kernel_size)
        self.num_experts = num_experts
        self.router_reduction = router_reduction
        self.num_groups = num_groups
        self.group_on_out = group_on_out

        if self.group_on_out:
            if self.out_channels % num_groups != 0:
                raise ValueError(
                    f"out_channels ({self.out_channels}) must be divisible by num_groups ({num_groups}) when group_on_out=True"
                )
        else:
            if self.in_channels % num_groups != 0:
                raise ValueError(
                    f"in_channels ({self.in_channels}) must be divisible by num_groups ({num_groups}) when group_on_out=False"
                )

        self.weight = nn.Parameter(
            torch.empty(num_experts, self.out_channels, self.in_channels)
        )
        if conv.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.empty(num_experts, self.out_channels))

        self.router = (
            Router(self.in_channels, num_experts * num_groups, router_reduction, nonlin, nonlin_kwargs)
            if use_internal_router
            else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        for expert_weight in self.weight:
            nn.init.kaiming_uniform_(expert_weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_channels)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, scores: torch.Tensor = None) -> torch.Tensor:
        if scores is None:
            if self.router is None:
                raise ValueError("shared router scores must be supplied to this CondPWConv")
            scores = self.router(x)

        batch_size = x.shape[0]

        # Sample-level routing: scores is of shape [batch_size, num_groups * num_experts]
        if scores.ndim == 2:
            if self.num_groups == 1:
                weight = torch.mm(scores, self.weight.flatten(1)).reshape(
                    batch_size, self.out_channels, self.in_channels
                )
                output = torch.bmm(weight, x.flatten(2))
                if self.bias is not None:
                    mixed_bias = torch.mm(scores, self.bias)
                    output = output + mixed_bias.unsqueeze(-1)
            else:
                N = batch_size
                G = self.num_groups
                E = self.num_experts
                scores_grouped = scores.reshape(N, G, E)
                if self.group_on_out:
                    C_out_g = self.out_channels // G
                    w_reshaped = self.weight.reshape(E, G, C_out_g, self.in_channels)
                    w_permuted = w_reshaped.permute(1, 0, 2, 3).reshape(G, E, -1)
                    blended = torch.matmul(scores_grouped.unsqueeze(2), w_permuted.unsqueeze(0))
                    weight = blended.squeeze(2).reshape(N, self.out_channels, self.in_channels)
                    if self.bias is not None:
                        b_reshaped = self.bias.reshape(E, G, C_out_g).permute(1, 0, 2)
                        blended_bias = torch.matmul(scores_grouped.unsqueeze(2), b_reshaped.unsqueeze(0))
                        mixed_bias = blended_bias.squeeze(2).reshape(N, self.out_channels)
                    else:
                        mixed_bias = None
                else:
                    C_in_g = self.in_channels // G
                    w_reshaped = self.weight.reshape(E, self.out_channels, G, C_in_g)
                    w_permuted = w_reshaped.permute(2, 0, 1, 3).reshape(G, E, -1)
                    blended = torch.matmul(scores_grouped.unsqueeze(2), w_permuted.unsqueeze(0))
                    weight = blended.squeeze(2).reshape(N, G, self.out_channels, C_in_g).permute(0, 2, 1, 3).reshape(N, self.out_channels, self.in_channels)
                    if self.bias is not None:
                        scores_mean = scores_grouped.mean(dim=1)
                        mixed_bias = torch.mm(scores_mean, self.bias)
                    else:
                        mixed_bias = None

                output = torch.bmm(weight, x.flatten(2))
                if mixed_bias is not None:
                    output = output + mixed_bias.unsqueeze(-1)

            return output.reshape(batch_size, self.out_channels, *x.shape[2:])

        # Region/Patch-level routing: scores is of shape [batch_size, *grid_shape, num_groups * num_experts]
        grid_shape = scores.shape[1:-1]
        if scores.shape[0] != batch_size or len(grid_shape) != self.spatial_dims:
            raise ValueError("router score grid does not match the convolution input")
        if any(size % grid for size, grid in zip(x.shape[2:], grid_shape)):
            raise ValueError(
                f"input spatial shape {tuple(x.shape[2:])} must be divisible by routing grid {tuple(grid_shape)}"
            )

        tile_shape = tuple(size // grid for size, grid in zip(x.shape[2:], grid_shape))
        split_shape = [batch_size, self.in_channels]
        for grid, tile in zip(grid_shape, tile_shape):
            split_shape.extend((grid, tile))
        grid_axes = list(range(2, 2 + 2 * self.spatial_dims, 2))
        tile_axes = list(range(3, 2 + 2 * self.spatial_dims, 2))
        tiled_input = (
            x.reshape(split_shape)
            .permute(0, *grid_axes, 1, *tile_axes)
            .reshape(-1, self.in_channels, math.prod(tile_shape))
        )

        flat_scores = scores.reshape(-1, self.num_experts * self.num_groups)

        if self.num_groups == 1:
            weight = torch.mm(flat_scores, self.weight.flatten(1)).reshape(
                -1, self.out_channels, self.in_channels
            )
            output = torch.bmm(weight, tiled_input)
            if self.bias is not None:
                output.add_(torch.mm(flat_scores, self.bias).unsqueeze(-1))
        else:
            N = flat_scores.shape[0]
            G = self.num_groups
            E = self.num_experts
            scores_grouped = flat_scores.reshape(N, G, E)
            if self.group_on_out:
                C_out_g = self.out_channels // G
                w_reshaped = self.weight.reshape(E, G, C_out_g, self.in_channels)
                w_permuted = w_reshaped.permute(1, 0, 2, 3).reshape(G, E, -1)
                blended = torch.matmul(scores_grouped.unsqueeze(2), w_permuted.unsqueeze(0))
                weight = blended.squeeze(2).reshape(N, self.out_channels, self.in_channels)
                if self.bias is not None:
                    b_reshaped = self.bias.reshape(E, G, C_out_g).permute(1, 0, 2)
                    blended_bias = torch.matmul(scores_grouped.unsqueeze(2), b_reshaped.unsqueeze(0))
                    mixed_bias = blended_bias.squeeze(2).reshape(N, self.out_channels)
                else:
                    mixed_bias = None
            else:
                C_in_g = self.in_channels // G
                w_reshaped = self.weight.reshape(E, self.out_channels, G, C_in_g)
                w_permuted = w_reshaped.permute(2, 0, 1, 3).reshape(G, E, -1)
                blended = torch.matmul(scores_grouped.unsqueeze(2), w_permuted.unsqueeze(0))
                weight = blended.squeeze(2).reshape(N, G, self.out_channels, C_in_g).permute(0, 2, 1, 3).reshape(N, self.out_channels, self.in_channels)
                if self.bias is not None:
                    scores_mean = scores_grouped.mean(dim=1)
                    mixed_bias = torch.mm(scores_mean, self.bias)
                else:
                    mixed_bias = None

            output = torch.bmm(weight, tiled_input)
            if mixed_bias is not None:
                output.add_(mixed_bias.unsqueeze(-1))

        tiled_shape = [batch_size, *grid_shape, self.out_channels, *tile_shape]
        channel_axis = 1 + self.spatial_dims
        spatial_axes = []
        for dim in range(self.spatial_dims):
            spatial_axes.extend((1 + dim, channel_axis + 1 + dim))
        return output.reshape(tiled_shape).permute(0, channel_axis, *spatial_axes).reshape(
            batch_size, self.out_channels, *x.shape[2:]
        )


def _expand_expansion_ratios(
    value: Union[float, Sequence[float]], n_stages: int, name: str
) -> List[float]:
    values = [value] * n_stages if isinstance(value, Real) else list(value)
    if len(values) != n_stages:
        raise ValueError(f"{name} must contain exactly {n_stages} values, got {len(values)}")
    if any(not isinstance(item, Real) or isinstance(item, bool) or item <= 0 for item in values):
        raise ValueError(f"{name} values must be positive numbers")
    return [float(item) for item in values]


def _expand_num_experts(value: Union[int, Sequence[int]], n_stages: int, name: str) -> List[int]:
    values = [value] * n_stages if isinstance(value, int) else list(value)
    if len(values) != n_stages:
        raise ValueError(f"{name} must contain exactly {n_stages} values, got {len(values)}")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in values):
        raise ValueError(f"{name} values must be non-negative integers")
    return values


def _expand_num_groups(value: Union[int, Sequence[int]], n_stages: int, name: str) -> List[int]:
    values = [value] * n_stages if isinstance(value, (int, np.integer)) else list(value)
    if len(values) != n_stages:
        raise ValueError(f"{name} must contain exactly {n_stages} values, got {len(values)}")
    if any(not isinstance(item, (int, np.integer)) or isinstance(item, bool) or item <= 0 for item in values):
        raise ValueError(f"{name} values must be positive integers greater than or equal to 1")
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


def _forward_routed_conv_block(
    conv_block: ConvDropoutNormReLU, x: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    for module in conv_block.all_modules:
        x = module(x, scores) if isinstance(module, CondPWConv) else module(x)
    return x


def _expertify_pointwise(
    conv_block: ConvDropoutNormReLU,
    num_experts: int,
    router_reduction: float,
    nonlin: Union[None, Type[nn.Module]],
    nonlin_kwargs: dict,
    num_groups: int = 1,
    group_on_out: bool = True,
) -> None:
    conv = conv_block.conv
    if conv.groups != 1 or any(kernel_size != 1 for kernel_size in conv.kernel_size):
        raise ValueError("CondMobileNet only expertifies dense pointwise convolutions")
    conditional_conv = CondPWConv(
        conv,
        num_experts,
        router_reduction,
        nonlin,
        nonlin_kwargs,
        use_internal_router=False,
        num_groups=num_groups,
        group_on_out=group_on_out,
    )
    conv_block.conv = conditional_conv
    conv_block.all_modules[0] = conditional_conv


class InvertedBottleneckBlock(nn.Module):
    """Inverted bottleneck with optional pointwise routing and squeeze-and-excitation.
    
    The basic underlying block format is: PW -> Norm/Act -> DW -> Norm/Act -> PW -> Norm.
    Bias is set to False for all these convolutions as they have a proceeding normalization layer.
    """

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
        expansion_ratio: float = 3.0,
        num_experts: int = 0,
        cc_mlp_reduction: float = 0.125,
        tile_size: Union[None, Sequence[int]] = None,
        se_mlp_reduction: float = 0.125,
        enable_se: bool = False,
        use_cc: bool = False,
        num_groups: int = 1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = maybe_convert_scalar_to_list(conv_op, stride)
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        norm_op_kwargs = {} if norm_op_kwargs is None else norm_op_kwargs
        nonlin_kwargs = {} if nonlin_kwargs is None else nonlin_kwargs

        self.expanded_channels = int(round(expansion_ratio * input_channels))
        if self.expanded_channels <= 0:
            raise ValueError(f"expansion_ratio must produce at least one channel, got {expansion_ratio}")

        # PW expansion: PW -> Norm/Act
        self.expand = ConvDropoutNormReLU(
            conv_op, input_channels, self.expanded_channels, 1, 1, False,
            norm_op, norm_op_kwargs, None, None, nonlin, nonlin_kwargs
        )
        
        # Depthwise DW: DW -> Norm/Act
        self.depthwise_conv = conv_op(
            self.expanded_channels,
            self.expanded_channels,
            kernel_size,
            self.stride,
            padding=_same_padding(kernel_size),
            groups=self.expanded_channels,
            bias=False,
        )
        self.depthwise_norm = norm_op(self.expanded_channels, **norm_op_kwargs) if norm_op is not None else nn.Identity()
        self.depthwise_nonlin = nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()
        
        # PW projection: PW -> Norm
        self.project = ConvDropoutNormReLU(
            conv_op, self.expanded_channels, output_channels, 1, 1, False,
            norm_op, norm_op_kwargs, None, None, None, None
        )
        self.add_identity = input_channels == output_channels and all(i == 1 for i in self.stride)

        self.num_experts = num_experts if use_cc else 0
        self.num_groups = num_groups if use_cc else 1
        if use_cc:
            if num_experts <= 0:
                raise ValueError(
                    f"CondConv is enabled (use_cc=True) but num_experts is {num_experts}. "
                    "num_experts must be greater than 0 to configure a dynamic mixture of experts."
                )
            if num_groups <= 0:
                raise ValueError("num_groups must be greater than 0")
            if self.expanded_channels % num_groups != 0:
                raise ValueError(
                    f"Expanded channels ({self.expanded_channels}) must be divisible by num_groups ({num_groups})"
                )

            self.router = Router(
                input_channels, num_experts * num_groups, cc_mlp_reduction, nonlin, nonlin_kwargs, tile_size
            )
            _expertify_pointwise(
                self.expand, num_experts, cc_mlp_reduction, nonlin, nonlin_kwargs,
                num_groups=num_groups, group_on_out=True
            )
            _expertify_pointwise(
                self.project, num_experts, cc_mlp_reduction, nonlin, nonlin_kwargs,
                num_groups=num_groups, group_on_out=False
            )
        else:
            self.router = None

        se_tile_size = (
            None
            if tile_size is None
            else tuple(max(1, tile // step) for tile, step in zip(tile_size, self.stride))
        )
        self.se = (
            SqueezeAndExcitationBlock(
                output_channels, se_mlp_reduction, nonlin, nonlin_kwargs, se_tile_size
            )
            if enable_se
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.router is not None:
            scores = self.router(x)
            x = _forward_routed_conv_block(self.expand, x, scores)
            x = self.depthwise_nonlin(self.depthwise_norm(self.depthwise_conv(x)))
            x = _forward_routed_conv_block(self.project, x, scores)
        else:
            x = self.expand(x)
            x = self.depthwise_nonlin(self.depthwise_norm(self.depthwise_conv(x)))
            x = self.project(x)
            
        x = self.se(x)
        if self.add_identity:
            x = x + residual
        return x

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), (
            "just give the image size without color/feature channels or batch channel. "
            "Do not give input_size=(b, c, x, y(, z)). Give input_size=(x, y(, z))!"
        )
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
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
        expansion_ratio: float = 3.0,
        num_experts: int = 0,
        cc_mlp_reduction: float = 0.125,
        tile_size: Union[None, Sequence[int]] = None,
        se_mlp_reduction: float = 0.125,
        se_config: List[bool] = None,
        cc_config: List[bool] = None,
        num_groups: int = 1,
    ):
        super().__init__()
        if n_blocks <= 0:
            raise ValueError("n_blocks must be greater than 0")
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)
        self.output_channels = output_channels

        if se_config is None:
            se_config = [False] * n_blocks
        if cc_config is None:
            cc_config = [False] * n_blocks

        if len(se_config) != n_blocks:
            raise ValueError(f"se_config length ({len(se_config)}) must match n_blocks ({n_blocks})")
        if len(cc_config) != n_blocks:
            raise ValueError(f"cc_config length ({len(cc_config)}) must match n_blocks ({n_blocks})")

        def make_block(in_channels: int, stride, block_tile_size, block_se: bool, block_cc: bool):
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
                expansion_ratio=expansion_ratio,
                num_experts=num_experts,
                cc_mlp_reduction=cc_mlp_reduction,
                tile_size=block_tile_size,
                se_mlp_reduction=se_mlp_reduction,
                enable_se=block_se,
                use_cc=block_cc,
                num_groups=num_groups,
            )

        self.blocks = nn.Sequential(
            make_block(
                input_channels,
                initial_stride,
                None
                if tile_size is None
                else tuple(tile * stride for tile, stride in zip(tile_size, self.initial_stride)),
                se_config[0],
                cc_config[0],
            ),
            *[make_block(output_channels, 1, tile_size, se_config[i], cc_config[i]) for i in range(1, n_blocks)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    def compute_conv_feature_map_size(self, input_size):
        output = self.blocks[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
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
        encoder_n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        encoder_expansion_ratio: Union[float, Sequence[float]] = 3.0,
        return_skips: bool = True,
        stem_channels: int = None,
        stem_kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        stem_stride: Union[int, List[int], Tuple[int, ...]] = 1,
        num_experts: Union[int, Sequence[int]] = 0,
        cc_mlp_reduction: float = 0.125,
        tile_size: Union[None, Sequence[int]] = None,
        se_mlp_reduction: float = 0.125,
        encoder_se: Union[bool, List[bool], List[List[bool]]] = False,
        encoder_cc: Union[bool, List[bool], List[List[bool]]] = False,
        num_groups: Union[int, Sequence[int]] = 1,
    ):
        super().__init__()
        if isinstance(features_per_stage, int):
            raise TypeError(
                f"features_per_stage must be explicitly provided as a sequence of integers, "
                f"not a single integer: {features_per_stage}"
            )
        features_per_stage = list(features_per_stage)
        if len(features_per_stage) != n_stages:
            raise ValueError(f"features_per_stage must contain exactly {n_stages} values")

        encoder_n_blocks_per_stage = (
            [encoder_n_blocks_per_stage] * n_stages
            if isinstance(encoder_n_blocks_per_stage, int)
            else list(encoder_n_blocks_per_stage)
        )
        strides = [strides] * n_stages if isinstance(strides, int) else list(strides)
        if len(encoder_n_blocks_per_stage) != n_stages:
            raise ValueError(f"encoder_n_blocks_per_stage must contain exactly {n_stages} values")
        if len(strides) != n_stages:
            raise ValueError(f"strides must contain exactly {n_stages} values")
        kernel_sizes = _normalize_kernel_sizes(
            conv_op, kernel_sizes, n_stages, [3] * convert_conv_op_to_dim(conv_op)
        )
        self.num_experts = _expand_num_experts(
            num_experts, n_stages, "encoder_num_experts"
        )
        self.num_groups = _expand_num_groups(
            num_groups, n_stages, "encoder_num_groups"
        )
        self.expansion_ratios = _expand_expansion_ratios(
            encoder_expansion_ratio, n_stages, "encoder_expansion_ratio"
        )

        # Generate structural configs for SE and CC blocks
        self.se_config = _expand_block_config(
            encoder_se, n_stages, encoder_n_blocks_per_stage, "encoder_se"
        )
        self.cc_config = _expand_block_config(
            encoder_cc, n_stages, encoder_n_blocks_per_stage, "encoder_cc"
        )

        # Validate that stages using CondConv have valid expert counts
        for stage_idx in range(n_stages):
            if any(self.cc_config[stage_idx]) and self.num_experts[stage_idx] <= 0:
                raise ValueError(
                    f"CondConv is enabled in encoder stage {stage_idx} (block-level configs: {self.cc_config[stage_idx]}), "
                    f"but num_experts for this stage is {self.num_experts[stage_idx]}. It must be greater than 0."
                )

        stem_channels = features_per_stage[0] if stem_channels is None else stem_channels
        self.stem_kernel_size = maybe_convert_scalar_to_list(conv_op, stem_kernel_size)
        self.stem_stride = maybe_convert_scalar_to_list(conv_op, stem_stride)
        if tile_size is not None:
            if len(tile_size) != convert_conv_op_to_dim(conv_op):
                raise ValueError(
                    f"tile_size must contain exactly {convert_conv_op_to_dim(conv_op)} values"
                )
            if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in tile_size):
                raise ValueError("tile_size values must be positive integers")
            current_tile_size = [
                max(1, tile // stride) for tile, stride in zip(tile_size, self.stem_stride)
            ]
        else:
            current_tile_size = None
        
        # Stem applies no non-linearity (only Conv + Norm)
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
        stage_tile_sizes = []
        stage_input_channels = stem_channels
        for stage_idx in range(n_stages):
            if current_tile_size is not None:
                stage_stride = maybe_convert_scalar_to_list(conv_op, strides[stage_idx])
                current_tile_size = [
                    max(1, tile // stride)
                    for tile, stride in zip(current_tile_size, stage_stride)
                ]
                stage_tile_sizes.append(tuple(current_tile_size))
            stages.append(
                StackedCondInvertedBottleneckBlocks(
                    encoder_n_blocks_per_stage[stage_idx],
                    conv_op,
                    stage_input_channels,
                    features_per_stage[stage_idx],
                    kernel_sizes[stage_idx],
                    strides[stage_idx],
                    norm_op,
                    norm_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    self.expansion_ratios[stage_idx],
                    self.num_experts[stage_idx],
                    cc_mlp_reduction,
                    current_tile_size,
                    se_mlp_reduction,
                    self.se_config[stage_idx],
                    self.cc_config[stage_idx],
                    num_groups=self.num_groups[stage_idx],
                )
            )
            stage_input_channels = features_per_stage[stage_idx]

        self.stages = nn.ModuleList(stages)
        self.stage_tile_sizes = None if tile_size is None else stage_tile_sizes
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, stride) for stride in strides]
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
        input_size = [i // j for i, j in zip(input_size, self.stem_stride)]
        for stage_idx, stage in enumerate(self.stages):
            output += stage.compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[stage_idx])]
        return output


class CondUNetDecoder(nn.Module):
    def __init__(
        self,
        encoder: CondUNetEncoder,
        num_classes: int,
        decoder_n_blocks_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        decoder_expansion_ratio: Union[float, Sequence[float]] = 3.0,
        num_experts: Union[int, Sequence[int]] = 0,
        cc_mlp_reduction: float = 0.125,
        linear_upsampling: bool = True,
        se_mlp_reduction: float = 0.125,
        decoder_se: Union[bool, List[bool], List[List[bool]]] = False,
        decoder_cc: Union[bool, List[bool], List[List[bool]]] = False,
        num_groups: Union[int, Sequence[int]] = 1,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.linear_upsampling = linear_upsampling
        self.mode = _interpolation_mode(encoder.conv_op)
        n_stages_encoder = len(encoder.output_channels)
        _validate_native_resolution_decoder(encoder.strides)
        decoder_n_blocks_per_stage = (
            [decoder_n_blocks_per_stage] * (n_stages_encoder - 1)
            if isinstance(decoder_n_blocks_per_stage, int)
            else list(decoder_n_blocks_per_stage)
        )
        if len(decoder_n_blocks_per_stage) != n_stages_encoder - 1:
            raise ValueError(
                f"decoder_n_blocks_per_stage must contain exactly {n_stages_encoder - 1} values"
            )
        self.num_experts = _expand_num_experts(
            num_experts, n_stages_encoder - 1, "decoder_num_experts"
        )
        self.num_groups = _expand_num_groups(
            num_groups, n_stages_encoder - 1, "decoder_num_groups"
        )
        self.expansion_ratios = _expand_expansion_ratios(
            decoder_expansion_ratio, n_stages_encoder - 1, "decoder_expansion_ratio"
        )

        # Generate structural configs for SE and CC blocks
        self.se_config = _expand_block_config(
            decoder_se, n_stages_encoder - 1, decoder_n_blocks_per_stage, "decoder_se"
        )
        self.cc_config = _expand_block_config(
            decoder_cc, n_stages_encoder - 1, decoder_n_blocks_per_stage, "decoder_cc"
        )

        # Validate that stages using CondConv have valid expert counts
        for stage_idx in range(n_stages_encoder - 1):
            if any(self.cc_config[stage_idx]) and self.num_experts[stage_idx] <= 0:
                raise ValueError(
                    f"CondConv is enabled in decoder stage {stage_idx} (block-level configs: {self.cc_config[stage_idx]}), "
                    f"but num_experts for this stage is {self.num_experts[stage_idx]}. It must be greater than 0."
                )

        stages = []
        upsamplers = []
        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            if linear_upsampling:
                upsamplers.append(nn.Identity())
                stage_input_channels = input_features_below + input_features_skip
            else:
                stride = encoder.strides[-s]
                upsamplers.append(
                    transpconv_op(
                        input_features_below,
                        input_features_skip,
                        stride,
                        stride,
                        bias=encoder.conv_bias,
                    )
                )
                stage_input_channels = 2 * input_features_skip
            target_stage_idx = n_stages_encoder - s - 1
            stages.append(
                StackedCondInvertedBottleneckBlocks(
                    decoder_n_blocks_per_stage[s - 1],
                    encoder.conv_op,
                    stage_input_channels,
                    input_features_skip,
                    encoder.kernel_sizes[target_stage_idx],
                    1,
                    encoder.norm_op,
                    encoder.norm_op_kwargs,
                    encoder.nonlin,
                    encoder.nonlin_kwargs,
                    self.expansion_ratios[s - 1],
                    self.num_experts[s - 1],
                    cc_mlp_reduction,
                    None
                    if encoder.stage_tile_sizes is None
                    else encoder.stage_tile_sizes[target_stage_idx],
                    se_mlp_reduction,
                    self.se_config[s - 1],
                    self.cc_config[s - 1],
                    num_groups=self.num_groups[s - 1],
                )
            )

        self.stages = nn.ModuleList(stages)
        self.upsamplers = nn.ModuleList(upsamplers)
        head_op = _convert_conv_op_to_transposed(encoder.conv_op)
        head_padding = _same_padding(encoder.stem_kernel_size)
        head_output_padding = _transpose_output_padding(
            encoder.stem_kernel_size, encoder.stem_stride, head_padding
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
        x = skips[-1]
        for stage_idx, stage in enumerate(self.stages):
            skip = skips[-(stage_idx + 2)]
            if self.linear_upsampling:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode=self.mode, align_corners=False
                )
            else:
                x = self.upsamplers[stage_idx](x)
            x = stage(torch.cat((x, skip), dim=1))
        seg_output = self.seg_layer(x)
        return [seg_output] if self.deep_supervision else seg_output

    def compute_conv_feature_map_size(self, input_size):
        native_input_size = input_size
        input_size = [i // j for i, j in zip(input_size, self.encoder.stem_stride)]
        skip_sizes = []
        for stride in self.encoder.strides[:-1]:
            input_size = [i // j for i, j in zip(input_size, stride)]
            skip_sizes.append(input_size)

        output = np.int64(0)
        for stage_idx, stage in enumerate(self.stages):
            skip_size = skip_sizes[-(stage_idx + 1)]
            output += stage.compute_conv_feature_map_size(skip_size)
            if not self.linear_upsampling:
                output += np.prod(
                    [self.encoder.output_channels[-(stage_idx + 2)], *skip_size],
                    dtype=np.int64,
                )
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
        stem_channels: int = None,
        stem_kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        stem_stride: Union[int, List[int], Tuple[int, ...]] = 1,
        encoder_num_experts: Union[int, List[int], Tuple[int, ...]] = 0,
        decoder_num_experts: Union[int, List[int], Tuple[int, ...]] = 0,
        linear_upsampling: bool = True,
        tile_size: Union[None, List[int], Tuple[int, ...]] = None,
        encoder_se: Union[bool, List[bool], List[List[bool]]] = False,
        decoder_se: Union[bool, List[bool], List[List[bool]]] = False,
        encoder_cc: Union[bool, List[bool], List[List[bool]]] = False,
        decoder_cc: Union[bool, List[bool], List[List[bool]]] = False,
        se_mlp_reduction: float = 0.125,
        cc_mlp_reduction: float = 0.125,
        encoder_num_groups: Union[int, List[int], Tuple[int, ...]] = 1,
        decoder_num_groups: Union[int, List[int], Tuple[int, ...]] = 1,
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

        encoder_expansion_ratios = _expand_expansion_ratios(
            encoder_expansion_ratio, n_stages, "encoder_expansion_ratio"
        )
        decoder_expansion_ratios = _expand_expansion_ratios(
            decoder_expansion_ratio, n_stages - 1, "decoder_expansion_ratio"
        )
        
        self.encoder = CondUNetEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            encoder_n_blocks_per_stage=encoder_n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            encoder_expansion_ratio=encoder_expansion_ratios,
            return_skips=True,
            stem_channels=stem_channels,
            stem_kernel_size=stem_kernel_size,
            stem_stride=stem_stride,
            num_experts=encoder_num_experts,
            cc_mlp_reduction=cc_mlp_reduction,
            tile_size=tile_size,
            se_mlp_reduction=se_mlp_reduction,
            encoder_se=encoder_se,
            encoder_cc=encoder_cc,
            num_groups=encoder_num_groups,
        )
        self.decoder = CondUNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            decoder_n_blocks_per_stage=decoder_n_blocks_per_stage,
            deep_supervision=deep_supervision,
            decoder_expansion_ratio=decoder_expansion_ratios,
            num_experts=decoder_num_experts,
            cc_mlp_reduction=cc_mlp_reduction,
            linear_upsampling=linear_upsampling,
            se_mlp_reduction=se_mlp_reduction,
            decoder_se=decoder_se,
            decoder_cc=decoder_cc,
            num_groups=decoder_num_groups,
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))

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
            for expert_weight in module.weight:
                nn.init.kaiming_normal_(expert_weight, a=1e-2)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        InitWeights_He(1e-2)(module)
        if (
            isinstance(module, InvertedBottleneckBlock)
            and module.add_identity
            and hasattr(module.project, "norm")
        ):
            if module.project.norm.weight is not None:
                nn.init.constant_(module.project.norm.weight, 0)
            if module.project.norm.bias is not None:
                nn.init.constant_(module.project.norm.bias, 0)