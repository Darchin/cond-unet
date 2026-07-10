import pytest
import torch
from torch import nn

from nnunetv2.training.network_architecture.cond_unet import (
    CondPWConv,
    CondUNet,
    TiledPoolMLP,
    _derive_grid_shape,
)


def _small_model(**overrides) -> CondUNet:
    kwargs = {
        "input_channels": 1,
        "n_stages": 3,
        "features_per_stage": [8, 16, 32],
        "conv_op": nn.Conv2d,
        "kernel_sizes": 3,
        "strides": [[1, 1], [2, 2], [2, 2]],
        "encoder_n_blocks_per_stage": [1, 2, 1],
        "num_classes": 2,
        "decoder_n_blocks_per_stage": [1, 2],
        "norm_op": nn.InstanceNorm2d,
        "nonlin": nn.ReLU,
    }
    kwargs.update(overrides)
    return CondUNet(**kwargs)


def _explicit_ungrouped_condconv(
    layer: CondPWConv, x: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    output = torch.empty(x.shape[0], layer.out_channels, *x.shape[2:])
    grid_shape = scores.shape[1:-1] if scores.ndim > 2 else (1,) * layer.spatial_dims
    tile_shape = tuple(size // grid for size, grid in zip(x.shape[2:], grid_shape))
    score_grid = scores.reshape(x.shape[0], *grid_shape, layer.num_experts)

    for batch_idx in range(x.shape[0]):
        for tile_idx in torch.cartesian_prod(*(torch.arange(size) for size in grid_shape)):
            tile_idx = tuple(int(index) for index in tile_idx.reshape(-1))
            tile_slices = tuple(
                slice(index * tile, (index + 1) * tile)
                for index, tile in zip(tile_idx, tile_shape)
            )
            tile_scores = score_grid[(batch_idx, *tile_idx)]
            weight = torch.einsum("e,eoi->oi", tile_scores, layer.weight)
            bias = None if layer.bias is None else torch.einsum("e,eo->o", tile_scores, layer.bias)
            tile_input = x[(batch_idx, slice(None), *tile_slices)].reshape(layer.in_channels, -1)
            tile_output = weight @ tile_input
            if bias is not None:
                tile_output += bias[:, None]
            output[(batch_idx, slice(None), *tile_slices)] = tile_output.reshape(
                layer.out_channels, *tile_shape
            )
    return output


def test_condpwconv_bmm_matches_explicit_global_and_tiled_mixtures():
    torch.manual_seed(0)
    layer = CondPWConv(
        nn.Conv2d(3, 4, 1, bias=True),
        num_experts=3,
        router_reduction=0.5,
        nonlin=nn.ReLU,
        use_internal_router=False,
    )
    x = torch.randn(2, 3, 8, 6)

    for scores in (torch.rand(2, 3), torch.rand(2, 2, 3, 3)):
        actual = layer(x, scores)
        expected = _explicit_ungrouped_condconv(layer, x, scores)
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("group_on_out", [True, False])
def test_grouped_expert_blending_matches_explicit_channel_group_mixture(group_on_out):
    layer = CondPWConv(
        nn.Conv2d(4, 6, 1, bias=True),
        num_experts=2,
        router_reduction=0.5,
        nonlin=nn.ReLU,
        use_internal_router=False,
        num_groups=2,
        group_on_out=group_on_out,
    )
    scores = torch.rand(3, 2, 2)
    actual_weight, actual_bias = layer._blend_experts(scores.flatten(1))

    expected_weight = torch.empty_like(actual_weight)
    if group_on_out:
        output_channels_per_group = layer.out_channels // layer.num_groups
        for group_idx in range(layer.num_groups):
            channel_slice = slice(
                group_idx * output_channels_per_group, (group_idx + 1) * output_channels_per_group
            )
            expected_weight[:, channel_slice] = torch.einsum(
                "be,eoi->boi", scores[:, group_idx], layer.weight[:, channel_slice]
            )
        expected_bias = torch.cat(
            [
                torch.einsum("be,eo->bo", scores[:, group_idx], bias_group)
                for group_idx, bias_group in enumerate(
                    layer.bias.split(output_channels_per_group, dim=1)
                )
            ],
            dim=1,
        )
    else:
        input_channels_per_group = layer.in_channels // layer.num_groups
        for group_idx in range(layer.num_groups):
            channel_slice = slice(
                group_idx * input_channels_per_group, (group_idx + 1) * input_channels_per_group
            )
            expected_weight[:, :, channel_slice] = torch.einsum(
                "be,eoi->boi", scores[:, group_idx], layer.weight[:, :, channel_slice]
            )
        expected_bias = torch.einsum("be,eo->bo", scores.mean(dim=1), layer.bias)

    torch.testing.assert_close(actual_weight, expected_weight)
    torch.testing.assert_close(actual_bias, expected_bias)


def test_tiled_pool_rejects_geometry_that_cannot_form_equal_tiles():
    pool = TiledPoolMLP(4, 2, 0.5, nn.ReLU, max_grid_size=3)
    with pytest.raises(ValueError, match="must be divisible by grid_shape"):
        pool(torch.randn(1, 4, 10, 10), grid_shape=(3, 3))


def test_grid_shape_tracks_bottleneck_aspect_ratio_and_uses_valid_divisors():
    assert _derive_grid_shape((6, 12, 12), max_grid_size=4) == (2, 4, 4)
    assert _derive_grid_shape((10, 12, 12), max_grid_size=4) == (2, 4, 4)


def test_tiled_addons_use_one_bottleneck_relative_grid_across_network():
    model = _small_model(
        se={
            "encoder": [False, True, True],
            "decoder": True,
            "max_grid_size": 4,
        },
        cc={
            "encoder": [False, True, True],
            "encoder_num_experts": 2,
            "max_grid_size": 4,
        }
    )
    pooled_output_shapes = []
    hooks = [
        module.register_forward_hook(
            lambda _module, _inputs, output: pooled_output_shapes.append(output.shape)
        )
        for module in model.modules()
        if isinstance(module, TiledPoolMLP) and module.max_grid_size is not None
    ]

    output = model(torch.randn(1, 1, 32, 64))
    for hook in hooks:
        hook.remove()

    assert output.shape == (1, 2, 32, 64)
    assert pooled_output_shapes
    assert all(tuple(shape[1:-1]) == (2, 4) for shape in pooled_output_shapes)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"deep_supervision": True}, "does not support deep supervision"),
        ({"upsample_mode": "typo"}, "upsample_mode must be"),
        ({"kernel_sizes": 2}, "kernel_sizes values must be odd"),
        ({"se": {"max_grid_size": 0}}, "must be a positive integer or None"),
        (
            {
                "strides": [[1, 1], [1, 2], [2, 2]],
                "se": {"max_grid_size": 2},
            },
            "requires isotropic encoder stage strides",
        ),
    ],
)
def test_invalid_public_architecture_options_fail_early(overrides, message):
    with pytest.raises(ValueError, match=message):
        _small_model(**overrides)


def test_per_stage_and_per_block_addon_configuration_builds_and_runs():
    model = _small_model(
        encoder_expansion_ratio=[1.0, 1.5, 2.0],
        decoder_expansion_ratio=[1.5, 1.0],
        se={"encoder": [[False], [True, False], [True]], "decoder": [[True], [False, True]]},
        cc={
            "encoder": [[False], [False, True], [False]],
            "decoder": [[True], [False, False]],
            "encoder_num_experts": [0, 2, 0],
            "decoder_num_experts": [2, 0],
        },
    )
    output = model(torch.randn(1, 1, 32, 32))
    assert output.shape == (1, 2, 32, 32)
