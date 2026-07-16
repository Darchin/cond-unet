import inspect

import pytest
import torch
from torch import nn

from nnunetv2.training.network_architecture.cond_unet import (
    CCConfig,
    CondPWConv,
    CondUNet,
    Router,
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
    outputs = []
    for sample, sample_scores in zip(x, scores):
        grouped_scores = sample_scores.reshape(layer.num_groups, layer.num_experts)
        grouped_weight = []
        for group_idx in range(layer.num_groups):
            weight = sum(
                grouped_scores[group_idx, expert_idx]
                * layer.expert_weights[expert_idx, group_idx]
                for expert_idx in range(layer.num_experts)
            )
            grouped_weight.append(weight)
        if layer.group_on_out:
            weight = torch.cat(grouped_weight, dim=0)
        else:
            weight = torch.cat(grouped_weight, dim=1)
        outputs.append(
            torch.nn.functional.conv2d(
                sample.unsqueeze(0), weight[:, :, None, None]
            ).squeeze(0)
        )
    return torch.stack(outputs)


@pytest.mark.parametrize("group_on_out", [True, False])
def test_condpwconv_matches_explicit_per_sample_mixture(group_on_out):
    layer = CondPWConv(
        nn.Conv2d(4, 6, 1, bias=False),
        num_experts=3,
        num_groups=2,
        group_on_out=group_on_out,
    )
    x = torch.randn(2, 4, 5, 7)
    scores = torch.softmax(torch.randn(2, 2, 3), dim=-1).flatten(1)

    actual = layer(x, scores)
    expected = _explicit_condconv(layer, x, scores)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("group_on_out", "expected_shape"),
    [(True, (3, 2, 3, 4)), (False, (3, 2, 6, 2))],
)
def test_condpwconv_stores_full_rank_grouped_expert_weights(
    group_on_out, expected_shape
):
    layer = CondPWConv(
        nn.Conv2d(4, 6, 1, bias=False),
        num_experts=3,
        num_groups=2,
        group_on_out=group_on_out,
    )
    assert layer.expert_weights.shape == expected_shape
    assert torch.count_nonzero(layer.expert_weights) > 0
    assert not hasattr(layer, "base_weight")
    assert not hasattr(layer, "left_factor")
    assert not hasattr(layer, "right_factor")


def test_condpwconv_rejects_bias():
    with pytest.raises(ValueError, match="bias-free"):
        CondPWConv(nn.Conv2d(3, 4, 1, bias=True), num_experts=2)


def test_condpwconv_rejects_tiled_scores():
    layer = CondPWConv(nn.Conv2d(3, 4, 1, bias=False), num_experts=2)
    with pytest.raises(ValueError, match="sample-level"):
        layer(torch.randn(1, 3, 4, 4), torch.randn(1, 2, 2, 2))


@pytest.mark.parametrize("group_on_out", [True, False])
def test_grouped_expert_blending_has_expected_shape(group_on_out):
    layer = CondPWConv(
        nn.Conv2d(4, 6, 1, bias=False),
        num_experts=3,
        num_groups=2,
        group_on_out=group_on_out,
    )
    output = layer(torch.randn(2, 4, 5, 5), torch.randn(2, 6))
    assert output.shape == (2, 6, 5, 5)


def test_router_is_bottlenecked_and_uniformly_initialized():
    router = Router(
        17,
        4,
        num_groups=3,
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
    assert router.output_projection.out_features == 12
    torch.testing.assert_close(
        router.output_projection.weight,
        torch.zeros_like(router.output_projection.weight),
    )

    grouped_bias = router.output_projection.bias.reshape(3, 4)
    torch.testing.assert_close(
        torch.nn.functional.softplus(grouped_bias),
        torch.full((3, 4), 0.25),
    )

    scores = router(torch.randn(2, 17, 4, 5)).reshape(2, 3, 4)
    torch.testing.assert_close(scores, torch.full_like(scores, 0.25))
    torch.testing.assert_close(scores.sum(dim=-1), torch.ones(2, 3))


def test_router_uses_softplus_normalized_mixture():
    router = Router(3, 3, num_groups=1, reduction=2, nonlin=nn.ReLU)
    with torch.no_grad():
        router.output_projection.weight.zero_()
        router.output_projection.bias.copy_(torch.tensor([-2.0, 0.0, 2.0]))
    raw_expert_scores = torch.nn.functional.softplus(torch.tensor([-2.0, 0.0, 2.0]))
    mixture = raw_expert_scores / raw_expert_scores.sum()
    actual = router(torch.randn(2, 3, 4, 5))
    torch.testing.assert_close(actual, mixture.expand(2, -1))
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(2))


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


def test_provided_normalization_is_used_throughout_model():
    model = _small_model(
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={"eps": 1e-3, "momentum": 0.2},
    )
    encoder_block = model.encoder.stages[0].blocks[0]
    decoder_block = model.decoder.stages[0].blocks[0]
    norms = [
        model.encoder.stem.convs[0].norm,
        encoder_block.expand.norm,
        encoder_block.depthwise.norm,
        encoder_block.project.norm,
        decoder_block.expand.norm,
        decoder_block.depthwise.norm,
        decoder_block.project.norm,
        model.decoder.seg_norm,
    ]
    assert all(isinstance(norm, nn.BatchNorm2d) for norm in norms)
    assert all(norm.eps == pytest.approx(1e-3) for norm in norms)
    assert all(norm.momentum == pytest.approx(0.2) for norm in norms)
    model(torch.randn(1, 1, 32, 32))


def test_missing_normalization_uses_no_op_behavior():
    model = _small_model(norm_op=None, norm_op_kwargs=None)
    block = model.encoder.stages[0].blocks[0]
    assert not hasattr(model.encoder.stem.convs[0], "norm")
    assert not hasattr(block.expand, "norm")
    assert isinstance(block.depthwise.norm, nn.Identity)
    assert not hasattr(block.project, "norm")
    assert isinstance(model.decoder.seg_norm, nn.Identity)
    model(torch.randn(1, 1, 32, 32))


def test_cc_group_counts_are_configured_per_encoder_and_decoder_stage():
    model = _small_model(
        cc={
            "encoder": True,
            "decoder": True,
            "encoder_num_experts": 2,
            "decoder_num_experts": 3,
            "encoder_num_groups": [2, 4, 8],
            "decoder_num_groups": [8, 4],
        }
    )
    assert model.encoder.num_groups == [2, 4, 8]
    assert model.decoder.num_groups == [8, 4]
    assert model.encoder.stages[1].blocks[0].expand.conv.num_groups == 4
    assert model.decoder.stages[0].blocks[0].expand.conv.num_groups == 8
    assert model.encoder.stages[1].blocks[0].router.output_projection.out_features == 8
    assert model.decoder.stages[0].blocks[0].router.output_projection.out_features == 24
    model(torch.randn(1, 1, 32, 32))


def test_cc_router_settings_are_propagated():
    model = _small_model(
        nonlin_kwargs={"inplace": True},
        cc={
            "encoder": [[True], [False, False], [False]],
            "encoder_num_experts": 2,
            "encoder_num_groups": 2,
            "reduction": 4.0,
        },
    )
    block = model.encoder.stages[0].blocks[0]
    assert block.router.hidden_channels == 2
    assert isinstance(block.router.nonlin, nn.ReLU)
    assert block.router.nonlin.inplace


@pytest.mark.parametrize("obsolete_key", ["se", "upsample_mode", "num_groups"])
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
        "encoder_num_groups",
        "decoder_num_groups",
        "reduction",
    }
    model_parameters = inspect.signature(CondUNet).parameters
    assert "se" not in model_parameters
    assert "upsample_mode" not in model_parameters
    assert "num_groups" not in model_parameters


@pytest.mark.parametrize("groups", [0, [1, 2]])
def test_cc_group_counts_are_validated(groups):
    with pytest.raises((ValueError, TypeError), match="encoder num_groups"):
        _small_model(cc={"encoder_num_groups": groups})


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
