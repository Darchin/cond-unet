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
        if layer.num_groups != 1:
            raise AssertionError("helper only supports ungrouped routing")
        weight = torch.einsum("e,eoi->oi", sample_scores, layer.weight)
        bias = (
            None
            if layer.bias is None
            else torch.einsum("e,eo->o", sample_scores, layer.bias)
        )
        outputs.append(
            torch.nn.functional.conv2d(
                sample.unsqueeze(0), weight[:, :, None, None], bias
            ).squeeze(0)
        )
    return torch.stack(outputs)


def test_condpwconv_matches_explicit_per_sample_mixture():
    layer = CondPWConv(nn.Conv2d(3, 4, 1, bias=True), num_experts=3)
    x = torch.randn(2, 3, 5, 7)
    scores = torch.sigmoid(torch.randn(2, 3))

    actual = layer(x, scores)
    expected = _explicit_condconv(layer, x, scores)

    torch.testing.assert_close(actual, expected)


def test_condpwconv_rejects_tiled_scores():
    layer = CondPWConv(nn.Conv2d(3, 4, 1), num_experts=2)
    with pytest.raises(ValueError, match="sample-level"):
        layer(torch.randn(1, 3, 4, 4), torch.randn(1, 2, 2, 2))


@pytest.mark.parametrize("group_on_out", [True, False])
def test_grouped_expert_blending_has_expected_shape(group_on_out):
    layer = CondPWConv(
        nn.Conv2d(4, 6, 1, bias=True),
        num_experts=3,
        num_groups=2,
        group_on_out=group_on_out,
    )
    output = layer(torch.randn(2, 4, 5, 5), torch.randn(2, 6))
    assert output.shape == (2, 6, 5, 5)


def test_router_is_direct_global_projection_with_sigmoid():
    router = Router(3, 4)
    assert list(router.children()) == [router.projection]
    assert isinstance(router.projection, nn.Linear)
    with torch.no_grad():
        router.projection.weight.fill_(0.25)
        router.projection.bias.zero_()
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    expected = torch.sigmoid(router.projection(x.mean(dim=(2, 3))))
    torch.testing.assert_close(router(x), expected)


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


@pytest.mark.parametrize("obsolete_key", ["se", "upsample_mode"])
def test_removed_top_level_architecture_keys_are_rejected(obsolete_key):
    with pytest.raises(TypeError, match=obsolete_key):
        _small_model(**{obsolete_key: None})


@pytest.mark.parametrize(
    "obsolete_key",
    [
        "reduction",
        "max_grid_size",
        "encoder_concat_global_context",
        "decoder_concat_global_context",
        "encoder_router_assignment",
        "decoder_router_assignment",
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
    }
    model_parameters = inspect.signature(CondUNet).parameters
    assert "se" not in model_parameters
    assert "upsample_mode" not in model_parameters


def test_cc_requires_experts_when_enabled():
    with pytest.raises(ValueError, match="num_experts is 0"):
        _small_model(cc={"encoder": True})
