import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import List, Optional, Sequence, Tuple, Type, Union

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
    values = [kernel_size] if isinstance(kernel_size, int) else list(kernel_size)
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0 for value in values):
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
    if any(not isinstance(item, (int, np.integer)) or isinstance(item, bool) or item <= 0 for item in values):
        raise ValueError(f"{name} values must be positive integers")
    if require_odd and any(item % 2 == 0 for item in values):
        raise ValueError(f"{name} values must be odd when using same padding")
    return [int(item) for item in values]


def _scale_tile_size(
    tile_size: Union[None, Sequence[int]], scale: Sequence[int], *, down: bool
) -> Optional[Tuple[int, ...]]:
    if tile_size is None:
        return None
    if down:
        return tuple(max(1, tile // factor) for tile, factor in zip(tile_size, scale))
    return tuple(tile * factor for tile, factor in zip(tile_size, scale))


def _tile_grid(spatial_shape: Sequence[int], tile_size: Sequence[int]) -> Tuple[int, ...]:
    grid_shape = tuple(
        max(1, spatial_size // requested_tile_size)
        for spatial_size, requested_tile_size in zip(spatial_shape, tile_size)
    )
    if any(spatial_size % grid_size for spatial_size, grid_size in zip(spatial_shape, grid_shape)):
        raise ValueError(
            f"input spatial shape {tuple(spatial_shape)} cannot be partitioned into equal routing/attention "
            f"tiles for tile_size={tuple(tile_size)} (derived grid={grid_shape})"
        )
    return grid_shape


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
        normalized = [_kernel_size_for_dim(default_kernel_size, dim) for _ in range(n_stages)]
        return [_normalize_spatial_param(conv_op, value, "kernel_sizes", require_odd=True) for value in normalized]
    if isinstance(kernel_sizes, int):
        normalized = [maybe_convert_scalar_to_list(conv_op, kernel_sizes) for _ in range(n_stages)]
        return [_normalize_spatial_param(conv_op, value, "kernel_sizes", require_odd=True) for value in normalized]

    kernel_sizes = list(kernel_sizes)
    if len(kernel_sizes) == 0:
        raise ValueError("kernel_sizes must not be empty")
    if isinstance(kernel_sizes[0], (list, tuple)):
        if len(kernel_sizes) != n_stages:
            raise ValueError(f"Expected one kernel size per stage ({n_stages}), got {len(kernel_sizes)}")
        normalized = [_kernel_size_for_dim(i, dim) for i in kernel_sizes]
        return [_normalize_spatial_param(conv_op, value, "kernel_sizes", require_odd=True) for value in normalized]
    if len(kernel_sizes) == dim:
        normalized = [_kernel_size_for_dim(kernel_sizes, dim) for _ in range(n_stages)]
        return [_normalize_spatial_param(conv_op, value, "kernel_sizes", require_odd=True) for value in normalized]
    if len(kernel_sizes) == n_stages:
        normalized = [maybe_convert_scalar_to_list(conv_op, int(i)) for i in kernel_sizes]
        return [_normalize_spatial_param(conv_op, value, "kernel_sizes", require_odd=True) for value in normalized]
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


class TiledPoolMLP(nn.Module):
    """Pool-then-MLP module with optional spatial tiling.

    When ``tile_size`` is None, performs global average pooling (standard behavior).
    When provided, the input is partitioned into a grid of tiles and each tile is
    pooled and processed independently, enabling spatially-varying attention/routing.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        reduction: float,
        nonlin: Union[None, Type[nn.Module]],
        nonlin_kwargs: dict = None,
        final_activation: Union[None, Type[nn.Module]] = nn.Sigmoid,
        tile_size: Union[None, Sequence[int]] = None,
    ):
        super().__init__()
        if reduction <= 0:
            raise ValueError("reduction must be greater than 0")
        if tile_size is not None and any(tile <= 0 for tile in tile_size):
            raise ValueError("tile_size values must be greater than 0")
        self.tile_size = None if tile_size is None else tuple(tile_size)
        hidden_channels = max(1, round(input_channels * reduction))
        nonlin_kwargs = {} if nonlin_kwargs is None else nonlin_kwargs
        activation = nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()
        layers = [
            nn.Linear(input_channels, hidden_channels),
            activation,
            nn.Linear(hidden_channels, output_channels),
        ]
        if final_activation is not None:
            layers.append(final_activation())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns pooled MLP output.

        - Global mode: shape ``[B, output_channels]``
        - Tiled mode: shape ``[B, *grid_shape, output_channels]``
        """
        spatial_dims = x.ndim - 2
        if self.tile_size is None:
            return self.mlp(x.mean(dim=tuple(range(2, x.ndim))))

        if len(self.tile_size) != spatial_dims:
            raise ValueError(
                f"tile_size has {len(self.tile_size)} dimensions, expected {spatial_dims}"
            )
        output_grid = _tile_grid(x.shape[2:], self.tile_size)
        adaptive_avg_pool = (
            F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d
        )[spatial_dims - 1]
        return self.mlp(adaptive_avg_pool(x, output_grid).movedim(1, -1))


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
        self.pool_mlp = TiledPoolMLP(
            channels, channels, reduction, nonlin, nonlin_kwargs,
            final_activation=nn.Sigmoid, tile_size=tile_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool_mlp(x)
        if self.pool_mlp.tile_size is None:
            spatial_dims = x.ndim - 2
            return x * scale.reshape(x.shape[0], x.shape[1], *([1] * spatial_dims))
        scale = scale.movedim(-1, 1)
        scale = F.interpolate(scale, size=x.shape[2:], mode="nearest")
        return x * scale


class Router(nn.Module):
    """Mean-only router that produces one expert score vector per spatial tile."""

    def __init__(
        self,
        input_channels: int,
        num_experts: int,
        reduction: float,
        nonlin: Union[None, Type[nn.Module]],
        nonlin_kwargs: dict = None,
        tile_size: Union[None, Sequence[int]] = None,
    ):
        super().__init__()
        self.pool_mlp = TiledPoolMLP(
            input_channels, num_experts, reduction, nonlin, nonlin_kwargs,
            final_activation=nn.Sigmoid, tile_size=tile_size,
        )
        self.tile_size = self.pool_mlp.tile_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_mlp(x)


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

    def _blend_experts(
        self, flat_scores: torch.Tensor
    ) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Blend expert weights and biases using routing scores.

        Args:
            flat_scores: Shape ``[N, num_experts * num_groups]`` where N is either
                the batch size (global routing) or batch_size * num_tiles (tiled routing).

        Returns:
            weight: ``[N, out_channels, in_channels]``
            bias: ``[N, out_channels]`` or None
        """
        N = flat_scores.shape[0]

        if self.num_groups == 1:
            weight = torch.mm(flat_scores, self.weight.flatten(1)).reshape(
                N, self.out_channels, self.in_channels
            )
            bias = torch.mm(flat_scores, self.bias) if self.bias is not None else None
            return weight, bias

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
                bias = blended_bias.squeeze(2).reshape(N, self.out_channels)
            else:
                bias = None
        else:
            C_in_g = self.in_channels // G
            w_reshaped = self.weight.reshape(E, self.out_channels, G, C_in_g)
            w_permuted = w_reshaped.permute(2, 0, 1, 3).reshape(G, E, -1)
            blended = torch.matmul(scores_grouped.unsqueeze(2), w_permuted.unsqueeze(0))
            weight = (
                blended.squeeze(2)
                .reshape(N, G, self.out_channels, C_in_g)
                .permute(0, 2, 1, 3)
                .reshape(N, self.out_channels, self.in_channels)
            )
            if self.bias is not None:
                scores_mean = scores_grouped.mean(dim=1)
                bias = torch.mm(scores_mean, self.bias)
            else:
                bias = None

        return weight, bias

    def _validate_scores(self, x: torch.Tensor, scores: torch.Tensor) -> None:
        expected_score_channels = self.num_experts * self.num_groups
        if scores.shape[0] != x.shape[0]:
            raise ValueError(
                f"router score batch size ({scores.shape[0]}) does not match input batch size ({x.shape[0]})"
            )
        if scores.shape[-1] != expected_score_channels:
            raise ValueError(
                f"router scores must have {expected_score_channels} values per sample/tile, "
                f"got {scores.shape[-1]}"
            )
        expected_ndim = self.spatial_dims + 2
        if scores.ndim not in (2, expected_ndim):
            raise ValueError(
                f"router scores must be sample-level (2D) or tiled ({expected_ndim}D), got {scores.ndim}D"
            )

    def _flatten_tiles(
        self, x: torch.Tensor, grid_shape: Sequence[int]
    ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        spatial_shape = x.shape[2:]
        if any(grid <= 0 for grid in grid_shape):
            raise ValueError(f"routing grid values must be positive, got {tuple(grid_shape)}")
        if any(size % grid for size, grid in zip(spatial_shape, grid_shape)):
            raise ValueError(
                f"input spatial shape {tuple(spatial_shape)} must be divisible by routing grid {tuple(grid_shape)}"
            )

        tile_shape = tuple(size // grid for size, grid in zip(spatial_shape, grid_shape))
        split_shape = [x.shape[0], self.in_channels]
        for grid, tile in zip(grid_shape, tile_shape):
            split_shape.extend((grid, tile))
        grid_axes = list(range(2, 2 + 2 * self.spatial_dims, 2))
        tile_axes = list(range(3, 2 + 2 * self.spatial_dims, 2))
        tiled_input = (
            x.reshape(split_shape)
            .permute(0, *grid_axes, 1, *tile_axes)
            .reshape(-1, self.in_channels, math.prod(tile_shape))
        )
        return tiled_input, tile_shape

    def _restore_tiles(
        self,
        output: torch.Tensor,
        batch_size: int,
        grid_shape: Sequence[int],
        tile_shape: Sequence[int],
    ) -> torch.Tensor:
        tiled_shape = [batch_size, *grid_shape, self.out_channels, *tile_shape]
        channel_axis = 1 + self.spatial_dims
        spatial_axes = []
        for dim in range(self.spatial_dims):
            spatial_axes.extend((1 + dim, channel_axis + 1 + dim))
        spatial_shape = tuple(grid * tile for grid, tile in zip(grid_shape, tile_shape))
        return output.reshape(tiled_shape).permute(0, channel_axis, *spatial_axes).reshape(
            batch_size, self.out_channels, *spatial_shape
        )

    def forward(self, x: torch.Tensor, scores: torch.Tensor = None) -> torch.Tensor:
        if scores is None:
            if self.router is None:
                raise ValueError("shared router scores must be supplied to this CondPWConv")
            scores = self.router(x)

        batch_size = x.shape[0]
        self._validate_scores(x, scores)

        # Sample-level routing: scores is [batch_size, num_groups * num_experts]
        if scores.ndim == 2:
            weight, bias = self._blend_experts(scores)
            output = torch.bmm(weight, x.flatten(2))
            if bias is not None:
                output = output + bias.unsqueeze(-1)
            return output.reshape(batch_size, self.out_channels, *x.shape[2:])

        # Region/Patch-level routing: scores is [batch_size, *grid_shape, num_groups * num_experts]
        grid_shape = scores.shape[1:-1]
        tiled_input, tile_shape = self._flatten_tiles(x, grid_shape)

        flat_scores = scores.reshape(-1, self.num_experts * self.num_groups)
        weight, bias = self._blend_experts(flat_scores)

        output = torch.bmm(weight, tiled_input)
        if bias is not None:
            output.add_(bias.unsqueeze(-1))

        return self._restore_tiles(output, batch_size, grid_shape, tile_shape)


def _expand_expansion_ratios(
    value: Union[float, Sequence[float]], n_stages: int, name: str
) -> List[float]:
    values = [value] * n_stages if isinstance(value, Real) else list(value)
    if len(values) != n_stages:
        raise ValueError(f"{name} must contain exactly {n_stages} values, got {len(values)}")
    if any(not isinstance(item, Real) or isinstance(item, bool) or item <= 0 for item in values):
        raise ValueError(f"{name} values must be positive numbers")
    return [float(item) for item in values]


def _expand_int_param(
    value: Union[int, Sequence[int]], n_stages: int, name: str, *, min_value: int = 0
) -> List[int]:
    """Expand a scalar-or-sequence of ints to a per-stage list, validating >= min_value."""
    values = [value] * n_stages if isinstance(value, (int, np.integer)) else list(value)
    if len(values) != n_stages:
        raise ValueError(f"{name} must contain exactly {n_stages} values, got {len(values)}")
    if any(not isinstance(item, (int, np.integer)) or isinstance(item, bool) or item < min_value for item in values):
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
    num_groups: int
    se_blocks: Tuple[bool, ...]
    cc_blocks: Tuple[bool, ...]


def _normalize_stage_settings(
    n_stages: int,
    n_blocks_per_stage: Union[int, Sequence[int]],
    expansion_ratio: Union[float, Sequence[float]],
    num_experts: Union[int, Sequence[int]],
    num_groups: Union[int, Sequence[int]],
    se: BoolConfig,
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
    group_counts = _expand_int_param(
        num_groups, n_stages, f"{context} num_groups", min_value=1
    )
    se_blocks = _expand_block_config(se, n_stages, block_counts, f"{context} se")
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
                num_groups=group_counts[stage_idx],
                se_blocks=tuple(se_blocks[stage_idx]),
                cc_blocks=tuple(cc_blocks[stage_idx]),
            )
        )
    return settings


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


class DepthwiseConvBlock(nn.Module):
    """Depthwise separable convolution block: DW Conv -> Norm -> Activation."""

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
    ):
        super().__init__()
        if not isinstance(channels, (int, np.integer)) or isinstance(channels, bool) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        kernel_size = _normalize_spatial_param(
            conv_op, kernel_size, "kernel_size", require_odd=True
        )
        stride = _normalize_spatial_param(conv_op, stride, "stride")
        norm_op_kwargs = {} if norm_op_kwargs is None else norm_op_kwargs
        nonlin_kwargs = {} if nonlin_kwargs is None else nonlin_kwargs
        self.conv = conv_op(
            channels, channels, kernel_size, stride,
            padding=_same_padding(kernel_size),
            groups=channels, bias=False,
        )
        self.norm = norm_op(channels, **norm_op_kwargs) if norm_op is not None else nn.Identity()
        self.nonlin = nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlin(self.norm(self.conv(x)))


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
        cc_reduction: float = 0.125,
        se_tile_size: Union[None, Sequence[int]] = None,
        cc_tile_size: Union[None, Sequence[int]] = None,
        se_reduction: float = 0.125,
        se: bool = False,
        cc: bool = False,
        num_groups: int = 1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = _normalize_spatial_param(conv_op, stride, "stride")
        kernel_size = _normalize_spatial_param(
            conv_op, kernel_size, "kernel_size", require_odd=True
        )
        se_tile_size = (
            None
            if se_tile_size is None
            else _normalize_spatial_param(conv_op, se_tile_size, "se_tile_size")
        )
        cc_tile_size = (
            None
            if cc_tile_size is None
            else _normalize_spatial_param(conv_op, cc_tile_size, "cc_tile_size")
        )
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
        self.depthwise = DepthwiseConvBlock(
            conv_op, self.expanded_channels, kernel_size, self.stride,
            norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
        )

        # PW projection: PW -> Norm
        self.project = ConvDropoutNormReLU(
            conv_op, self.expanded_channels, output_channels, 1, 1, False,
            norm_op, norm_op_kwargs, None, None, None, None
        )
        self.add_identity = input_channels == output_channels and all(i == 1 for i in self.stride)

        self.num_experts = num_experts if cc else 0
        self.num_groups = num_groups if cc else 1
        if cc:
            if num_experts <= 0:
                raise ValueError(
                    f"CondConv is enabled (cc=True) but num_experts is {num_experts}. "
                    "num_experts must be greater than 0 to configure a dynamic mixture of experts."
                )
            if num_groups <= 0:
                raise ValueError("num_groups must be greater than 0")
            if self.expanded_channels % num_groups != 0:
                raise ValueError(
                    f"Expanded channels ({self.expanded_channels}) must be divisible by num_groups ({num_groups})"
                )

            self.router = Router(
                input_channels, num_experts * num_groups, cc_reduction, nonlin, nonlin_kwargs, cc_tile_size
            )
            _expertify_pointwise(
                self.expand, num_experts, cc_reduction, nonlin, nonlin_kwargs,
                num_groups=num_groups, group_on_out=True
            )
            _expertify_pointwise(
                self.project, num_experts, cc_reduction, nonlin, nonlin_kwargs,
                num_groups=num_groups, group_on_out=False
            )
        else:
            self.router = None

        se_tile_size_scaled = _scale_tile_size(se_tile_size, self.stride, down=True)
        self.se_block = (
            SqueezeAndExcitationBlock(
                output_channels, se_reduction, nonlin, nonlin_kwargs, se_tile_size_scaled
            )
            if se
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.router is not None:
            scores = self.router(x)
            x = _forward_routed_conv_block(self.expand, x, scores)
            x = self.depthwise(x)
            x = _forward_routed_conv_block(self.project, x, scores)
        else:
            x = self.expand(x)
            x = self.depthwise(x)
            x = self.project(x)

        x = self.se_block(x)
        if self.add_identity:
            x = x + residual
        return x

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), (
            "just give the image size without color/feature channels or batch channel. "
            "Do not give input_size=(b, c, x, y(, z)). Give input_size=(x, y(, z))!"
        )
        size_after_stride = _conv_output_shape(input_size, self.depthwise.conv.kernel_size, self.stride)
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
        cc_reduction: float = 0.125,
        se_tile_size: Union[None, Sequence[int]] = None,
        cc_tile_size: Union[None, Sequence[int]] = None,
        se_reduction: float = 0.125,
        se_config: List[bool] = None,
        cc_config: List[bool] = None,
        num_groups: int = 1,
    ):
        super().__init__()
        if not isinstance(n_blocks, (int, np.integer)) or isinstance(n_blocks, bool) or n_blocks <= 0:
            raise ValueError("n_blocks must be greater than 0")
        self.initial_stride = _normalize_spatial_param(conv_op, initial_stride, "initial_stride")
        self.output_channels = output_channels

        if se_config is None:
            se_config = [False] * n_blocks
        if cc_config is None:
            cc_config = [False] * n_blocks

        if len(se_config) != n_blocks:
            raise ValueError(f"se_config length ({len(se_config)}) must match n_blocks ({n_blocks})")
        if len(cc_config) != n_blocks:
            raise ValueError(f"cc_config length ({len(cc_config)}) must match n_blocks ({n_blocks})")

        def make_block(in_channels: int, stride, block_se_tile_size, block_cc_tile_size, block_se: bool, block_cc: bool):
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
                cc_reduction=cc_reduction,
                se_tile_size=block_se_tile_size,
                cc_tile_size=block_cc_tile_size,
                se_reduction=se_reduction,
                se=block_se,
                cc=block_cc,
                num_groups=num_groups,
            )

        self.blocks = nn.Sequential(
            make_block(
                input_channels,
                initial_stride,
                _scale_tile_size(se_tile_size, self.initial_stride, down=False),
                _scale_tile_size(cc_tile_size, self.initial_stride, down=False),
                se_config[0],
                cc_config[0],
            ),
            *[make_block(output_channels, 1, se_tile_size, cc_tile_size, se_config[i], cc_config[i]) for i in range(1, n_blocks)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

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
        cc_reduction: float = 0.125,
        se_tile_size: Union[None, Sequence[int]] = None,
        cc_tile_size: Union[None, Sequence[int]] = None,
        se_reduction: float = 0.125,
        se: BoolConfig = False,
        cc: BoolConfig = False,
        num_groups: Union[int, Sequence[int]] = 1,
    ):
        super().__init__()
        if not isinstance(n_stages, (int, np.integer)) or isinstance(n_stages, bool) or n_stages <= 0:
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
            raise ValueError(f"features_per_stage must contain exactly {n_stages} values")
        if any(
            not isinstance(channels, (int, np.integer)) or isinstance(channels, bool) or channels <= 0
            for channels in features_per_stage
        ):
            raise ValueError("features_per_stage values must be positive integers")
        features_per_stage = [int(channels) for channels in features_per_stage]

        raw_strides = [strides] * n_stages if isinstance(strides, int) else list(strides)
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
            num_groups,
            se,
            cc,
            "encoder",
        )
        self.num_experts = [settings.num_experts for settings in stage_settings]
        self.num_groups = [settings.num_groups for settings in stage_settings]
        self.expansion_ratios = [settings.expansion_ratio for settings in stage_settings]
        self.se_config = [list(settings.se_blocks) for settings in stage_settings]
        self.cc_config = [list(settings.cc_blocks) for settings in stage_settings]

        stem_channels = features_per_stage[0] if stem_channels is None else stem_channels
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

        normalized_se_tile_size = (
            None
            if se_tile_size is None
            else _normalize_spatial_param(conv_op, se_tile_size, "se.tile_size")
        )
        normalized_cc_tile_size = (
            None
            if cc_tile_size is None
            else _normalize_spatial_param(conv_op, cc_tile_size, "cc.tile_size")
        )
        current_se_tile_size = _scale_tile_size(normalized_se_tile_size, self.stem_stride, down=True)
        current_cc_tile_size = _scale_tile_size(normalized_cc_tile_size, self.stem_stride, down=True)

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
        stage_se_tile_sizes = []
        stage_cc_tile_sizes = []
        stage_input_channels = stem_channels
        for stage_idx, settings in enumerate(stage_settings):
            stage_stride = strides[stage_idx]
            current_se_tile_size = _scale_tile_size(current_se_tile_size, stage_stride, down=True)
            current_cc_tile_size = _scale_tile_size(current_cc_tile_size, stage_stride, down=True)
            if current_se_tile_size is not None:
                stage_se_tile_sizes.append(current_se_tile_size)
            if current_cc_tile_size is not None:
                stage_cc_tile_sizes.append(current_cc_tile_size)

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
                    expansion_ratio=settings.expansion_ratio,
                    num_experts=settings.num_experts,
                    cc_reduction=cc_reduction,
                    se_tile_size=current_se_tile_size,
                    cc_tile_size=current_cc_tile_size,
                    se_reduction=se_reduction,
                    se_config=list(settings.se_blocks),
                    cc_config=list(settings.cc_blocks),
                    num_groups=settings.num_groups,
                )
            )
            stage_input_channels = features_per_stage[stage_idx]

        self.stages = nn.ModuleList(stages)
        self.stage_se_tile_sizes = None if se_tile_size is None else stage_se_tile_sizes
        self.stage_cc_tile_sizes = None if cc_tile_size is None else stage_cc_tile_sizes
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
        input_size = _conv_output_shape(input_size, self.stem_kernel_size, self.stem_stride)
        for stage_idx, stage in enumerate(self.stages):
            output += stage.compute_conv_feature_map_size(input_size)
            input_size = _conv_output_shape(
                input_size, self.kernel_sizes[stage_idx], self.strides[stage_idx]
            )
        return output


class CondUNetDecoder(nn.Module):
    def __init__(
        self,
        encoder: CondUNetEncoder,
        num_classes: int,
        n_blocks_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        expansion_ratio: Union[float, Sequence[float]] = 3.0,
        num_experts: Union[int, Sequence[int]] = 0,
        cc_reduction: float = 0.125,
        upsample_mode: str = "linear",
        se_reduction: float = 0.125,
        se: BoolConfig = False,
        cc: BoolConfig = False,
        num_groups: Union[int, Sequence[int]] = 1,
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
            raise ValueError("CondUNet does not support deep supervision; set deep_supervision=False")
        if upsample_mode not in {"linear", "transposed"}:
            raise ValueError(
                f"upsample_mode must be 'linear' or 'transposed', got {upsample_mode!r}"
            )
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.upsample_mode = upsample_mode
        self.interp_mode = _interpolation_mode(encoder.conv_op)
        n_stages_encoder = len(encoder.output_channels)
        _validate_native_resolution_decoder(encoder.strides)
        stage_settings = _normalize_stage_settings(
            n_stages_encoder - 1,
            n_blocks_per_stage,
            expansion_ratio,
            num_experts,
            num_groups,
            se,
            cc,
            "decoder",
        )
        self.num_experts = [settings.num_experts for settings in stage_settings]
        self.num_groups = [settings.num_groups for settings in stage_settings]
        self.expansion_ratios = [settings.expansion_ratio for settings in stage_settings]
        self.se_config = [list(settings.se_blocks) for settings in stage_settings]
        self.cc_config = [list(settings.cc_blocks) for settings in stage_settings]

        stages = []
        upsamplers = []
        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        use_linear = upsample_mode == "linear"
        for s in range(1, n_stages_encoder):
            settings = stage_settings[s - 1]
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            if use_linear:
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
                    expansion_ratio=settings.expansion_ratio,
                    num_experts=settings.num_experts,
                    cc_reduction=cc_reduction,
                    se_tile_size=None
                    if encoder.stage_se_tile_sizes is None
                    else encoder.stage_se_tile_sizes[target_stage_idx],
                    cc_tile_size=None
                    if encoder.stage_cc_tile_sizes is None
                    else encoder.stage_cc_tile_sizes[target_stage_idx],
                    se_reduction=se_reduction,
                    se_config=list(settings.se_blocks),
                    cc_config=list(settings.cc_blocks),
                    num_groups=settings.num_groups,
                )
            )

        self.stages = nn.ModuleList(stages)
        self.upsamplers = nn.ModuleList(upsamplers)
        head_op = get_matching_convtransp(conv_op=encoder.conv_op)
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
        if self.deep_supervision:
            raise RuntimeError("CondUNet does not support deep supervision")
        x = skips[-1]
        for stage_idx, stage in enumerate(self.stages):
            skip = skips[-(stage_idx + 2)]
            if self.upsample_mode == "linear":
                x = F.interpolate(
                    x, size=skip.shape[2:], mode=self.interp_mode, align_corners=False
                )
            else:
                x = self.upsamplers[stage_idx](x)
            x = stage(torch.cat((x, skip), dim=1))
        seg_output = self.seg_layer(x)
        return seg_output

    def compute_conv_feature_map_size(self, input_size):
        native_input_size = input_size
        input_size = _conv_output_shape(
            input_size, self.encoder.stem_kernel_size, self.encoder.stem_stride
        )
        skip_sizes = []
        for kernel_size, stride in zip(self.encoder.kernel_sizes[:-1], self.encoder.strides[:-1]):
            input_size = _conv_output_shape(input_size, kernel_size, stride)
            skip_sizes.append(input_size)

        output = np.int64(0)
        for stage_idx, stage in enumerate(self.stages):
            skip_size = skip_sizes[-(stage_idx + 1)]
            output += stage.compute_conv_feature_map_size(skip_size)
            if self.upsample_mode != "linear":
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
        upsample_mode: str = "linear",
        stem: Union[StemConfig, dict, None] = None,
        se: Union[SEConfig, dict, None] = None,
        cc: Union[CCConfig, dict, None] = None,
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
        se = _normalize_config(se, SEConfig)
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
            num_experts=cc.encoder_num_experts,
            cc_reduction=cc.reduction,
            se_tile_size=se.tile_size,
            cc_tile_size=cc.tile_size,
            se_reduction=se.reduction,
            se=se.encoder,
            cc=cc.encoder,
            num_groups=cc.encoder_num_groups,
        )
        self.decoder = CondUNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_blocks_per_stage=decoder_n_blocks_per_stage,
            deep_supervision=deep_supervision,
            expansion_ratio=decoder_expansion_ratio,
            num_experts=cc.decoder_num_experts,
            cc_reduction=cc.reduction,
            upsample_mode=upsample_mode,
            se_reduction=se.reduction,
            se=se.decoder,
            cc=cc.decoder,
            num_groups=cc.decoder_num_groups,
        )

    def forward(self, x: torch.Tensor):
        input_spatial_shape = x.shape[2:]
        output = self.decoder(self.encoder(x))
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
            for expert_weight in module.weight:
                nn.init.kaiming_normal_(expert_weight, a=1e-2)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
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
