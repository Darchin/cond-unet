import inspect
import itertools

import pytest
import torch
from torch import nn

from nnunetv2.training.network_architecture.cond_unet import (
    CCConfig,
    CondPWConv,
    CondUNet,
    InvertedBottleneckBlock,
    LayerNorm,
    Router,
    SEConfig,
    SqueezeExcitation,
    layer_norm,
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


def _explicit_condconv(
    layer: CondPWConv, x: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    output = torch.empty(x.shape[0], layer.out_channels, *x.shape[2:])
    grid_size = scores.shape[1:-1]
    tile_shape = tuple(size // grid for size, grid in zip(x.shape[2:], grid_size))
    for batch_idx in range(x.shape[0]):
        for tile_idx in itertools.product(*(range(size) for size in grid_size)):
            tile_slices = tuple(
                slice(index * tile, (index + 1) * tile)
                for index, tile in zip(tile_idx, tile_shape)
            )
            weight = torch.einsum(
                "e,eoi->oi",
                scores[(batch_idx, *tile_idx)],
                layer.expert_weights,
            )
            tile_input = x[(batch_idx, slice(None), *tile_slices)].flatten(1)
            tile_output = weight @ tile_input
            output[(batch_idx, slice(None), *tile_slices)] = tile_output.reshape(
                layer.out_channels, *tile_shape
            )
    return output


@pytest.mark.parametrize("grid_size", [(1, 1), (2, 3)])
def test_condpwconv_matches_explicit_global_and_tiled_mixture(grid_size):
    layer = CondPWConv(nn.Conv2d(4, 6, 1, bias=False), num_experts=3)
    x = torch.randn(2, 4, 6, 9)
    scores = torch.softmax(torch.randn(2, *grid_size, 3), dim=-1)

    actual = layer(x, scores)
    expected = _explicit_condconv(layer, x, scores)

    torch.testing.assert_close(actual, expected)


def test_condpwconv_matches_explicit_tiled_3d_mixture():
    layer = CondPWConv(nn.Conv3d(3, 4, 1, bias=False), num_experts=2)
    x = torch.randn(1, 3, 4, 6, 8)
    scores = torch.softmax(torch.randn(1, 2, 3, 2, 2), dim=-1)

    actual = layer(x, scores)
    expected = _explicit_condconv(layer, x, scores)

    torch.testing.assert_close(actual, expected)


def test_condpwconv_stores_full_rank_expert_weights():
    layer = CondPWConv(nn.Conv2d(4, 6, 1, bias=False), num_experts=3)
    assert layer.expert_weights.shape == (3, 6, 4)
    assert torch.count_nonzero(layer.expert_weights) > 0
    assert not hasattr(layer, "base_weight")
    assert not hasattr(layer, "left_factor")
    assert not hasattr(layer, "right_factor")


def test_condpwconv_rejects_bias():
    with pytest.raises(ValueError, match="bias-free"):
        CondPWConv(nn.Conv2d(3, 4, 1, bias=True), num_experts=2)


def test_condpwconv_rejects_non_divisible_tiles():
    layer = CondPWConv(nn.Conv2d(3, 4, 1, bias=False), num_experts=2)
    with pytest.raises(ValueError, match="must be divisible by routing grid"):
        layer(torch.randn(1, 3, 5, 4), torch.randn(1, 2, 2, 2))


def test_tiled_expert_blending_has_expected_shape():
    layer = CondPWConv(nn.Conv2d(4, 6, 1, bias=False), num_experts=3)
    output = layer(torch.randn(2, 4, 6, 10), torch.randn(2, 2, 5, 3))
    assert output.shape == (2, 6, 6, 10)


def test_router_is_bottlenecked_and_uniformly_initialized():
    router = Router(
        17,
        4,
        grid_size=(2, 3),
        reduction=8,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 0.2, "inplace": True},
    )
    assert router.hidden_channels == 2
    assert router.input_projection.in_features == 17
    assert router.input_projection.out_features == 2
    assert isinstance(router.nonlin, nn.LeakyReLU)
    assert router.nonlin.negative_slope == pytest.approx(0.2)
    assert router.nonlin.inplace
    assert router.output_projection.out_features == 4
    torch.testing.assert_close(
        router.output_projection.weight,
        torch.zeros_like(router.output_projection.weight),
    )

    torch.testing.assert_close(
        torch.nn.functional.softplus(router.output_projection.bias),
        torch.full((4,), 0.25),
    )

    scores = router(torch.randn(2, 17, 4, 6))
    assert scores.shape == (2, 2, 3, 4)
    torch.testing.assert_close(scores, torch.full_like(scores, 0.25))
    torch.testing.assert_close(scores.sum(dim=-1), torch.ones(2, 2, 3))


def test_router_uses_sigmoid_normalized_mixture():
    router = Router(3, 3, grid_size=None, reduction=2, nonlin=nn.ReLU)
    with torch.no_grad():
        router.output_projection.weight.zero_()
        router.output_projection.bias.copy_(torch.tensor([-2.0, 0.0, 2.0]))
    raw_expert_scores = torch.sigmoid(torch.tensor([-2.0, 0.0, 2.0]))
    mixture = raw_expert_scores / raw_expert_scores.sum()
    actual = router(torch.randn(2, 3, 4, 5))
    assert actual.shape == (2, 1, 1, 3)
    torch.testing.assert_close(actual, mixture.expand(2, 1, 1, -1))
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(2, 1, 1))


def test_router_uses_adaptive_average_pooling_for_tile_descriptors():
    router = Router(2, 3, grid_size=(2, 3), reduction=2, nonlin=nn.ReLU)
    descriptors = []
    handle = router.input_projection.register_forward_pre_hook(
        lambda _module, inputs: descriptors.append(inputs[0].detach().clone())
    )
    x = torch.arange(48, dtype=torch.float32).reshape(1, 2, 4, 6)
    try:
        router(x)
    finally:
        handle.remove()
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (2, 3)).movedim(1, -1)
    torch.testing.assert_close(descriptors[0], expected)


def test_each_cc_block_has_an_independent_router():
    model = _small_model(
        cc={
            "encoder": [[False], [True, True], [False]],
            "encoder_num_experts": [0, 2, 0],
        }
    )
    blocks = model.encoder.stages[1].blocks
    assert isinstance(blocks[0].router, Router)
    assert isinstance(blocks[1].router, Router)
    assert blocks[0].router is not blocks[1].router


def test_squeeze_excitation_is_bottlenecked_and_identity_initialized():
    se = SqueezeExcitation(
        17,
        grid_size=(2, 3),
        reduction=8,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 0.2, "inplace": True},
    )
    assert se.hidden_channels == 2
    assert se.input_projection.in_features == 17
    assert se.input_projection.out_features == 2
    assert se.output_projection.out_features == 17
    assert isinstance(se.nonlin, nn.LeakyReLU)
    torch.testing.assert_close(
        se.output_projection.weight,
        torch.zeros_like(se.output_projection.weight),
    )
    torch.testing.assert_close(
        se.output_projection.bias,
        torch.zeros_like(se.output_projection.bias),
    )

    x = torch.randn(2, 17, 4, 6)
    torch.testing.assert_close(se(x), x)


def test_tiled_squeeze_excitation_linearly_interpolates_scores():
    se = SqueezeExcitation(2, grid_size=(2, 2), reduction=2, nonlin=nn.Identity)
    with torch.no_grad():
        se.input_projection.weight.fill_(1)
        se.input_projection.bias.zero_()
        se.output_projection.weight.copy_(torch.tensor([[0.01], [-0.01]]))
        se.output_projection.bias.zero_()
    x = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    descriptor = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2)).movedim(1, -1)
    logits = se.output_projection(se.input_projection(descriptor))
    expected_scores = torch.nn.functional.interpolate(
        (2 * torch.sigmoid(logits)).movedim(-1, 1),
        size=x.shape[2:],
        mode="bilinear",
        align_corners=False,
    )

    torch.testing.assert_close(se(x), x * expected_scores)
    assert torch.all(expected_scores > 0)
    assert torch.all(expected_scores < 2)


def test_squeeze_excitation_rejects_non_divisible_grid():
    se = SqueezeExcitation(3, grid_size=(2, 2), nonlin=nn.ReLU)
    with pytest.raises(ValueError, match="must be divisible by SE grid"):
        se(torch.randn(1, 3, 5, 4))


def test_tiled_squeeze_excitation_supports_3d_features():
    se = SqueezeExcitation(3, grid_size=(2, 3, 2), nonlin=nn.ReLU)
    x = torch.randn(1, 3, 4, 6, 8)
    torch.testing.assert_close(se(x), x)


@pytest.mark.parametrize(
    ("placement", "expected_order"),
    [
        ("start", ["pre_depthwise", "expand", "se", "depthwise", "project"]),
        ("middle", ["pre_depthwise", "expand", "depthwise", "se", "project"]),
        ("end", ["pre_depthwise", "expand", "depthwise", "project", "se"]),
    ],
)
def test_se_placement_within_inverted_bottleneck(placement, expected_order):
    model = _small_model(
        se={
            "encoder": [[True], [False, False], [False]],
            "placement": placement,
        }
    )
    block = model.encoder.stages[0].blocks[0]
    calls = []
    handles = [
        block.pre_depthwise.register_forward_hook(
            lambda _module, _inputs, _output: calls.append("pre_depthwise")
        ),
        block.expand.register_forward_hook(
            lambda _module, _inputs, _output: calls.append("expand")
        ),
        block.depthwise.register_forward_hook(
            lambda _module, _inputs, _output: calls.append("depthwise")
        ),
        block.se.register_forward_pre_hook(
            lambda _module, _inputs: calls.append("se")
        ),
        block.project.norm.register_forward_hook(
            lambda _module, _inputs, _output: calls.append("project")
        ),
    ]
    try:
        model(torch.randn(1, 1, 32, 32))
    finally:
        for handle in handles:
            handle.remove()

    assert calls == expected_order
    expected_channels = (
        block.output_channels if placement == "end" else block.expanded_channels
    )
    assert block.se.input_projection.in_features == expected_channels


def test_inverted_bottleneck_is_depthwise_first_and_strides_on_second_depthwise():
    model = _small_model()
    block = model.encoder.stages[1].blocks[0]

    assert block.pre_depthwise.conv.in_channels == block.input_channels
    assert block.pre_depthwise.conv.groups == block.input_channels
    assert block.pre_depthwise.conv.kernel_size == block.depthwise.conv.kernel_size
    assert block.pre_depthwise.conv.stride == (1, 1)
    assert block.expand.conv.stride == (1, 1)
    assert block.depthwise.conv.stride == (2, 2)
    assert block.project.conv.stride == (1, 1)
    assert isinstance(block.pre_depthwise.norm, nn.Identity)
    assert isinstance(block.pre_depthwise.nonlin, nn.Identity)
    assert isinstance(block.depthwise.norm, nn.Identity)
    assert isinstance(block.depthwise.nonlin, nn.Identity)


@pytest.mark.parametrize(
    ("pre_norm", "expected_order"),
    [
        (
            False,
            [
                "pre_depthwise",
                "expand_conv",
                "expand_norm",
                "act",
                "depthwise",
                "project_conv",
                "project_norm",
            ],
        ),
        (
            True,
            [
                "pre_expand_norm",
                "pre_depthwise",
                "expand_conv",
                "act",
                "pre_project_norm",
                "depthwise",
                "project_conv",
            ],
        ),
    ],
)
def test_inverted_bottleneck_normalization_order(pre_norm, expected_order):
    block = InvertedBottleneckBlock(
        conv_op=nn.Conv2d,
        input_channels=4,
        output_channels=4,
        kernel_size=3,
        stride=1,
        norm_op=nn.BatchNorm2d,
        nonlin=nn.ReLU,
        pre_norm=pre_norm,
    )
    calls = []
    named_modules = {
        "pre_expand_norm": block.pre_expand_norm,
        "pre_depthwise": block.pre_depthwise.conv,
        "expand_conv": block.expand.conv,
        "act": block.expand.nonlin,
        "pre_project_norm": block.pre_project_norm,
        "depthwise": block.depthwise.conv,
        "project_conv": block.project.conv,
    }
    if hasattr(block.expand, "norm"):
        named_modules["expand_norm"] = block.expand.norm
    if hasattr(block.project, "norm"):
        named_modules["project_norm"] = block.project.norm
    handles = [
        module.register_forward_hook(
            lambda _module, _inputs, _output, name=name: calls.append(name)
        )
        for name, module in named_modules.items()
        if not isinstance(module, nn.Identity)
    ]
    try:
        block(torch.randn(2, 4, 8, 8))
    finally:
        for handle in handles:
            handle.remove()

    assert calls == expected_order


def test_condunet_pre_norm_is_applied_to_all_ib_blocks():
    model = _small_model(pre_norm=True)
    blocks = [
        block
        for stage in (*model.encoder.stages, *model.decoder.stages)
        for block in stage.blocks
    ]

    assert blocks
    assert all(block.pre_norm for block in blocks)
    assert all(isinstance(block.pre_expand_norm, nn.InstanceNorm2d) for block in blocks)
    assert all(isinstance(block.pre_project_norm, nn.InstanceNorm2d) for block in blocks)
    assert all(not hasattr(block.expand, "norm") for block in blocks)
    assert all(not hasattr(block.project, "norm") for block in blocks)
    assert model(torch.randn(1, 1, 32, 32)).shape == (1, 2, 32, 32)


def test_pre_norm_must_be_bool():
    with pytest.raises(TypeError, match="pre_norm must be a bool"):
        _small_model(pre_norm=1)


def test_one_block_reuses_its_router_scores_for_both_pointwise_convolutions():
    model = _small_model(
        cc={
            "encoder": [[True], [False, False], [False]],
            "encoder_num_experts": [2, 0, 0],
        }
    )
    block = model.encoder.stages[0].blocks[0]
    routed_scores = []
    handles = [
        module.register_forward_pre_hook(
            lambda _module, inputs: routed_scores.append(inputs[1])
        )
        for module in (block.expand.conv, block.project.conv)
    ]
    try:
        model(torch.randn(2, 1, 32, 32))
    finally:
        for handle in handles:
            handle.remove()
    assert len(routed_scores) == 2
    assert routed_scores[0] is routed_scores[1]


def test_model_forward_and_feature_map_accounting():
    model = _small_model()
    output = model(torch.randn(2, 1, 32, 48))
    assert output.shape == (2, 2, 32, 48)
    assert model.compute_conv_feature_map_size((32, 48)) > 0


def test_decoder_stages_have_no_transposed_convolutions():
    model = _small_model(stem={"stride": 2, "kernel_size": 3})
    assert not any(
        isinstance(module, nn.ConvTranspose2d)
        for stage in model.decoder.stages
        for module in stage.modules()
    )
    assert isinstance(model.decoder.seg_layer, nn.ConvTranspose2d)


@pytest.mark.parametrize("shape", [(2, 4, 9), (2, 4, 5, 7), (2, 4, 3, 5, 7)])
def test_channel_first_layer_norm_matches_pytorch_reference(shape):
    x = torch.randn(shape, dtype=torch.float64)
    norm = LayerNorm(shape[1], eps=1e-4).double()

    actual = norm(x)
    expected = torch.nn.functional.layer_norm(
        x.movedim(1, -1),
        (shape[1],),
        norm.weight,
        norm.bias,
        norm.eps,
    ).movedim(-1, 1)

    torch.testing.assert_close(actual, expected)


def test_layer_norm_functional_and_non_affine_module_match():
    x = torch.randn(2, 4, 5, 7)
    norm = LayerNorm(num_features=4, affine=False)

    assert norm.weight is None
    assert norm.bias is None
    torch.testing.assert_close(norm(x), layer_norm(x))


@pytest.mark.parametrize(
    ("norm_op", "norm_op_kwargs", "expected_eps"),
    [
        (nn.BatchNorm2d, {"eps": 1e-3, "momentum": 0.2}, 1e-3),
        (LayerNorm, {"eps": 1e-4, "affine": False}, 1e-4),
    ],
)
def test_model_uses_configured_normalization(
    norm_op, norm_op_kwargs, expected_eps
):
    model = _small_model(
        norm_op=norm_op,
        norm_op_kwargs=norm_op_kwargs,
    )
    encoder_block = model.encoder.stages[0].blocks[0]
    decoder_block = model.decoder.stages[0].blocks[0]
    norms = [
        model.encoder.stem.convs[0].norm,
        encoder_block.expand.norm,
        encoder_block.project.norm,
        decoder_block.expand.norm,
        decoder_block.project.norm,
        model.decoder.seg_norm,
    ]
    assert all(isinstance(norm, norm_op) for norm in norms)
    assert all(norm.eps == pytest.approx(expected_eps) for norm in norms)
    if norm_op is LayerNorm:
        assert all(not norm.affine for norm in norms)
    else:
        assert all(norm.momentum == pytest.approx(0.2) for norm in norms)
    model(torch.randn(1, 1, 32, 32))


def test_cc_grid_sizes_are_configured_per_encoder_and_decoder_stage():
    model = _small_model(
        cc={
            "encoder": True,
            "decoder": True,
            "encoder_num_experts": 2,
            "decoder_num_experts": 3,
            "encoder_grid_size": [[1, 1], [2, 2], [1, 2]],
            "decoder_grid_size": [[2, 2], [4, 2]],
        }
    )
    assert model.encoder.grid_sizes == [(1, 1), (2, 2), (1, 2)]
    assert model.decoder.grid_sizes == [(2, 2), (4, 2)]
    assert model.encoder.stages[1].blocks[0].grid_size == (2, 2)
    assert model.decoder.stages[0].blocks[0].grid_size == (2, 2)
    assert model.encoder.stages[1].blocks[0].router.output_projection.out_features == 2
    assert model.decoder.stages[0].blocks[0].router.output_projection.out_features == 3
    model(torch.randn(1, 1, 32, 32))


def test_one_grid_shape_is_broadcast_to_every_stage():
    model = _small_model(
        cc={
            "encoder_grid_size": [2, 1],
            "decoder_grid_size": [2, 2],
        }
    )
    assert model.encoder.grid_sizes == [(2, 1)] * 3
    assert model.decoder.grid_sizes == [(2, 2)] * 2


def test_none_grid_sizes_are_global_for_all_or_individual_stages():
    model = _small_model(
        cc={"encoder_grid_size": None},
        se={
            "encoder": True,
            "decoder": True,
            "encoder_grid_size": [None, [2, 2], None],
            "decoder_grid_size": None,
        },
    )
    assert model.encoder.grid_sizes == [(1, 1)] * 3
    assert model.encoder.se_grid_sizes == [(1, 1), (2, 2), (1, 1)]
    assert model.decoder.se_grid_sizes == [(1, 1)] * 2
    assert model.encoder.stages[1].blocks[0].se.grid_size == (2, 2)
    model(torch.randn(1, 1, 32, 32))


def test_strided_cc_block_validates_projection_feature_map_divisibility():
    model = _small_model(
        cc={
            "encoder": [[False], [True, False], [False]],
            "encoder_num_experts": [0, 2, 0],
            "encoder_grid_size": [[1, 1], [32, 1], [1, 1]],
        }
    )
    with pytest.raises(ValueError, match=r"spatial shape \(16, 16\).*routing grid"):
        model(torch.randn(1, 1, 32, 32))


def test_tiled_cc_model_backward_populates_router_and_expert_gradients():
    model = _small_model(
        cc={
            "encoder": [[False], [True, False], [False]],
            "encoder_num_experts": [0, 2, 0],
            "encoder_grid_size": [[1, 1], [2, 2], [1, 1]],
        }
    )
    block = model.encoder.stages[1].blocks[0]

    model(torch.randn(1, 1, 32, 32)).mean().backward()

    assert block.router.output_projection.weight.grad is not None
    assert block.expand.conv.expert_weights.grad is not None
    assert block.project.conv.expert_weights.grad is not None


def test_tiled_se_and_cc_model_backward_populates_addon_gradients():
    model = _small_model(
        cc={
            "encoder": [[False], [True, False], [False]],
            "encoder_num_experts": [0, 2, 0],
            "encoder_grid_size": [[1, 1], [2, 2], [1, 1]],
        },
        se={
            "encoder": [[False], [True, False], [False]],
            "encoder_grid_size": [[1, 1], [2, 2], [1, 1]],
        },
    )
    block = model.encoder.stages[1].blocks[0]

    model(torch.randn(1, 1, 32, 32)).mean().backward()

    assert block.router.output_projection.weight.grad is not None
    assert block.se.output_projection.weight.grad is not None
    assert block.expand.conv.expert_weights.grad is not None
    assert block.project.conv.expert_weights.grad is not None


def test_condunet_initializer_handles_ungrouped_expert_weights():
    model = _small_model(
        cc={
            "encoder": True,
            "encoder_num_experts": 4,
        }
    )

    model.apply(model.initialize)

    for module in model.modules():
        if isinstance(module, CondPWConv):
            assert module.expert_weights.shape[0] == 4
            assert torch.count_nonzero(module.expert_weights) > 0


def test_cc_router_settings_are_propagated():
    model = _small_model(
        nonlin_kwargs={"inplace": True},
        cc={
            "encoder": [[True], [False, False], [False]],
            "encoder_num_experts": 2,
            "encoder_grid_size": [2, 1],
            "reduction": 4.0,
        },
    )
    block = model.encoder.stages[0].blocks[0]
    assert block.router.hidden_channels == 2
    assert block.router.grid_size == (2, 1)
    assert isinstance(block.router.nonlin, nn.ReLU)
    assert block.router.nonlin.inplace


@pytest.mark.parametrize("obsolete_key", ["upsample_mode", "num_groups"])
def test_removed_top_level_architecture_keys_are_rejected(obsolete_key):
    with pytest.raises(TypeError, match=obsolete_key):
        _small_model(**{obsolete_key: None})


@pytest.mark.parametrize(
    "obsolete_key",
    [
        "max_grid_size",
        "encoder_concat_global_context",
        "decoder_concat_global_context",
        "encoder_router_assignment",
        "decoder_router_assignment",
        "delta_scale",
        "encoder_rank",
        "decoder_rank",
        "encoder_num_groups",
        "decoder_num_groups",
    ],
)
def test_removed_cc_keys_are_rejected(obsolete_key):
    with pytest.raises(TypeError, match=obsolete_key):
        _small_model(cc={obsolete_key: None})


def test_public_config_and_model_signatures_exclude_removed_parameters():
    cc_fields = set(CCConfig.__dataclass_fields__)
    assert cc_fields == {
        "encoder",
        "decoder",
        "encoder_num_experts",
        "decoder_num_experts",
        "encoder_grid_size",
        "decoder_grid_size",
        "reduction",
    }
    se_fields = set(SEConfig.__dataclass_fields__)
    assert se_fields == {
        "encoder",
        "decoder",
        "encoder_grid_size",
        "decoder_grid_size",
        "reduction",
        "placement",
    }
    assert CCConfig().encoder_grid_size is None
    assert CCConfig().decoder_grid_size is None
    assert SEConfig().encoder_grid_size is None
    assert SEConfig().decoder_grid_size is None
    assert SEConfig().placement == "middle"
    model_parameters = inspect.signature(CondUNet).parameters
    assert "se" in model_parameters
    assert model_parameters["pre_norm"].default is False
    assert "upsample_mode" not in model_parameters
    assert "num_groups" not in model_parameters


@pytest.mark.parametrize(
    ("grid_size", "error"),
    [
        (1, "explicit"),
        (0, "explicit"),
        ([1, 2, 3], "grid shape"),
        ([[1, 1], [2, 2]], "exactly 3"),
        ([[1, 1], [2, 0], [1, 1]], "positive"),
    ],
)
def test_cc_grid_sizes_are_validated(grid_size, error):
    with pytest.raises((ValueError, TypeError), match=error):
        _small_model(cc={"encoder_grid_size": grid_size})


def test_cc_requires_experts_when_enabled():
    with pytest.raises(ValueError, match="num_experts is 0"):
        _small_model(cc={"encoder": True})


@pytest.mark.parametrize(
    ("config", "error"),
    [
        ({"reduction": 0.5}, "reduction"),
        ({"reduction": float("inf")}, "reduction"),
    ],
)
def test_cc_router_settings_are_validated(config, error):
    with pytest.raises(ValueError, match=error):
        _small_model(cc=config)


@pytest.mark.parametrize(
    ("config", "error"),
    [
        ({"reduction": 0.5}, "reduction"),
        ({"reduction": float("inf")}, "reduction"),
    ],
)
def test_se_settings_are_validated(config, error):
    with pytest.raises(ValueError, match=error):
        _small_model(se=config)


@pytest.mark.parametrize("placement", [None, "before", 1])
def test_se_placement_is_validated(placement):
    with pytest.raises(ValueError, match="se.placement"):
        _small_model(se={"placement": placement})
