import pytest
import torch
from torch import nn

from nnunetv2.training.network_architecture.cond_unet import CondPWConv, CondUNet, TiledPoolMLP


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
    pool = TiledPoolMLP(4, 2, 0.5, nn.ReLU, tile_size=(3, 3))
    with pytest.raises(ValueError, match="cannot be partitioned into equal"):
        pool(torch.randn(1, 4, 10, 10))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"deep_supervision": True}, "does not support deep supervision"),
        ({"upsample_mode": "typo"}, "upsample_mode must be"),
        ({"kernel_sizes": 2}, "kernel_sizes values must be odd"),
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
