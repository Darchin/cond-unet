import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nnunetv2.training.network_architecture.cond_unet import (
    CCConfig,
    CondPWConv,
    CondUNet,
    CondUNetDecoder,
    InvertedBottleneckBlock,
    Router,
)
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerAdamW import (
    nnUNetTrainerAdamW,
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


def _explicit_grouped_condconv(
    layer: CondPWConv, x: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    output = []
    for batch_idx in range(x.shape[0]):
        if layer.group_axis == "output":
            expert_weights = layer.expert_weights.reshape(
                layer.num_experts,
                layer.num_groups,
                layer.out_channels // layer.num_groups,
                layer.in_channels,
            )
            grouped_weights = [
                torch.einsum(
                    "k,koi->oi", scores[batch_idx, group_idx], expert_weights[:, group_idx]
                )
                for group_idx in range(layer.num_groups)
            ]
            weight = torch.cat(grouped_weights, dim=0)
        else:
            expert_weights = layer.expert_weights.reshape(
                layer.num_experts,
                layer.out_channels,
                layer.num_groups,
                layer.in_channels // layer.num_groups,
            )
            grouped_weights = [
                torch.einsum(
                    "k,koj->oj", scores[batch_idx, group_idx], expert_weights[:, :, group_idx]
                )
                for group_idx in range(layer.num_groups)
            ]
            weight = torch.cat(grouped_weights, dim=1)
        output.append((weight @ x[batch_idx].flatten(1)).reshape(
            layer.out_channels, *x.shape[2:]
        ))
    return torch.stack(output)


@pytest.mark.parametrize(
    ("group_axis", "in_channels", "out_channels", "num_groups"),
    [
        ("output", 4, 6, 1),
        ("input", 6, 4, 1),
        ("output", 4, 6, 2),
        ("input", 6, 4, 2),
    ],
)
def test_condpwconv_matches_explicit_grouped_mixture(
    group_axis, in_channels, out_channels, num_groups
):
    layer = CondPWConv(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        num_experts=3,
        num_groups=num_groups,
        group_axis=group_axis,
    )
    x = torch.randn(2, in_channels, 5, 7, requires_grad=True)
    scores = torch.softmax(torch.randn(2, num_groups, 3), dim=-1)

    actual = layer(x, scores)
    expected = _explicit_grouped_condconv(layer, x, scores)

    torch.testing.assert_close(actual, expected)
    actual.mean().backward()
    assert x.grad is not None
    assert layer.expert_weights.grad is not None


def test_condpwconv_stores_full_rank_expert_weights():
    layer = CondPWConv(
        nn.Conv3d(4, 6, 1, bias=False),
        num_experts=3,
        num_groups=2,
        group_axis="output",
    )
    assert layer.expert_weights.shape == (3, 6, 4)
    assert torch.count_nonzero(layer.expert_weights) > 0


def test_condpwconv_rejects_bias_and_invalid_grouping():
    with pytest.raises(ValueError, match="bias-free"):
        CondPWConv(nn.Conv2d(3, 4, 1, bias=True), num_experts=2)
    with pytest.raises(ValueError, match="output_channels.*divisible"):
        CondPWConv(
            nn.Conv2d(4, 6, 1, bias=False),
            num_experts=2,
            num_groups=4,
            group_axis="output",
        )
    with pytest.raises(ValueError, match="input_channels.*divisible"):
        CondPWConv(
            nn.Conv2d(6, 4, 1, bias=False),
            num_experts=2,
            num_groups=4,
            group_axis="input",
        )


def test_router_direct_projection_is_zero_initialized_and_uniform():
    router = Router(5, 3, num_groups=2)
    assert router.hidden_channels is None
    assert router.input_projection is None
    assert router.output_projection.in_features == 5
    assert router.output_projection.out_features == 6
    torch.testing.assert_close(
        router.output_projection.weight,
        torch.zeros_like(router.output_projection.weight),
    )
    scores = router(torch.randn(4, 5, 3, 7))
    assert scores.shape == (4, 2, 3)
    torch.testing.assert_close(scores, torch.full_like(scores, 1 / 3))


def test_router_std_mean_descriptor_and_scaled_bottleneck():
    router = Router(
        5,
        3,
        num_groups=2,
        reduction=2,
        use_std=True,
        nonlin=nn.ReLU,
    )
    descriptors = []
    handle = router.input_projection.register_forward_pre_hook(
        lambda _module, inputs: descriptors.append(inputs[0].detach().clone())
    )
    x = torch.arange(40, dtype=torch.float32).reshape(2, 5, 2, 2)
    try:
        scores = router(x)
    finally:
        handle.remove()
    std, mean = torch.std_mean(x, dim=(2, 3), correction=0)
    torch.testing.assert_close(descriptors[0], torch.cat((std, mean), dim=1))
    assert router.descriptor_channels == 10
    assert router.hidden_channels == 5
    assert router.output_projection.out_features == 6
    assert scores.shape == (2, 2, 3)


def test_router_population_std_is_finite_for_singleton_spatial_shape():
    router = Router(3, 2, use_std=True)
    scores = router(torch.randn(2, 3, 1, 1))
    assert torch.isfinite(scores).all()


def test_expert_dropout_is_groupwise_normalized_and_training_only():
    router = Router(4, 5, num_groups=3, expert_dropout=0.8)
    x = torch.randn(64, 4, 3, 3)

    router.train()
    torch.manual_seed(0)
    scores = router(x)
    torch.testing.assert_close(scores.sum(-1), torch.ones(64, 3))
    assert torch.all(torch.count_nonzero(scores, dim=-1) >= 1)
    assert torch.count_nonzero(scores == 0) > 0

    router.eval()
    scores = router(x)
    torch.testing.assert_close(scores, torch.full_like(scores, 0.2))


@pytest.mark.parametrize("expert_dropout", [0.0, 0.5])
def test_all_stage_cc_routing_has_no_torch_compile_graph_breaks(expert_dropout):
    model = CondUNet(
        input_channels=1,
        n_stages=3,
        features_per_stage=[4, 8, 16],
        conv_op=nn.Conv2d,
        kernel_sizes=3,
        strides=[[1, 1], [2, 2], [2, 2]],
        encoder_n_blocks_per_stage=[1, 1, 1],
        num_classes=2,
        decoder_n_blocks_per_stage=[1, 1],
        norm_op=nn.InstanceNorm2d,
        nonlin=nn.ReLU,
        cc={
            "enabled": True,
            "num_experts": 4,
            "expert_dropout": expert_dropout,
        },
    ).train()

    explanation = torch._dynamo.explain(model)(torch.randn(2, 1, 16, 16))

    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    assert explanation.break_reasons == []


@pytest.mark.parametrize("rate", [-0.1, 1.0, float("inf"), True])
def test_expert_dropout_rejects_invalid_rates(rate):
    with pytest.raises(ValueError, match="expert_dropout"):
        Router(4, 2, expert_dropout=rate)


def test_each_cc_block_has_an_independent_router():
    model = _small_model(
        cc={
            "enabled": [[False], [True, True], [False]],
            "num_experts": [0, 2, 0],
        }
    )
    blocks = model.encoder.stages[1].blocks
    assert isinstance(blocks[0].router, Router)
    assert isinstance(blocks[1].router, Router)
    assert blocks[0].router is not blocks[1].router


def test_one_block_reuses_grouped_router_scores_for_both_pointwise_convolutions():
    model = _small_model(
        cc={
            "enabled": [[True], [False, False], [False]],
            "num_experts": [2, 0, 0],
            "num_groups": 2,
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
    assert routed_scores[0].shape == (2, 2, 2)
    assert block.expand.conv.group_axis == "output"
    assert block.project.conv.group_axis == "input"


def test_cc_options_propagate_and_model_backward_populates_gradients():
    model = _small_model(
        nonlin_kwargs={"inplace": True},
        cc={
            "enabled": [[False], [True, False], [False]],
            "num_experts": [0, 3, 0],
            "reduction": 4,
            "use_std": True,
            "expert_dropout": 0.2,
            "num_groups": 4,
        },
    )
    block = model.encoder.stages[1].blocks[0]
    assert block.router.descriptor_channels == 16
    assert block.router.hidden_channels == 4
    assert block.router.num_groups == 4
    assert block.router.expert_dropout == pytest.approx(0.2)

    model(torch.randn(2, 1, 32, 32)).mean().backward()
    assert block.router.output_projection.weight.grad is not None
    assert block.expand.conv.expert_weights.grad is not None
    assert block.project.conv.expert_weights.grad is not None


def test_num_groups_requires_divisible_expanded_channels():
    with pytest.raises(ValueError, match="output_channels.*divisible"):
        _small_model(
            cc={
                "enabled": [[True], [False, False], [False]],
                "num_experts": 2,
                "num_groups": 5,
            }
        )


def test_depthwise_dropout_is_after_activation_in_encoder_and_decoder():
    model = _small_model(
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.25},
    )
    encoder_block = model.encoder.stages[0].blocks[0]
    decoder_block = model.decoder.stages[0].blocks[0]
    assert isinstance(encoder_block.depthwise.dropout, nn.Dropout2d)
    assert isinstance(decoder_block.depthwise.dropout, nn.Dropout2d)
    assert encoder_block.depthwise.dropout.p == pytest.approx(0.25)

    calls = []
    handles = [
        encoder_block.depthwise.nonlin.register_forward_hook(
            lambda *_args: calls.append("activation")
        ),
        encoder_block.depthwise.dropout.register_forward_pre_hook(
            lambda *_args: calls.append("dropout")
        ),
    ]
    try:
        model(torch.randn(1, 1, 32, 32))
    finally:
        for handle in handles:
            handle.remove()
    assert calls == ["activation", "dropout"]


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


def test_missing_normalization_uses_no_op_behavior():
    model = _small_model(norm_op=None, norm_op_kwargs=None)
    block = model.encoder.stages[0].blocks[0]
    assert isinstance(block.depthwise.norm, nn.Identity)
    assert isinstance(model.decoder.seg_norm, nn.Identity)
    model(torch.randn(1, 1, 32, 32))


def test_encoder_stochastic_depth_uses_eligible_block_schedule():
    model = _small_model(
        encoder_n_blocks_per_stage=[3, 2, 1],
        drop_rate=0.3,
    )
    rates = [[block.drop_rate for block in stage.blocks] for stage in model.encoder.stages]
    for actual, expected in zip(rates, [[0.0, 0.1, 0.2], [0.0, 0.3], [0.0]]):
        assert actual == pytest.approx(expected)
    assert all(
        block.drop_rate == 0.0
        for stage in model.decoder.stages
        for block in stage.blocks
    )


def test_stochastic_depth_is_per_sample_scaled_and_training_only():
    block = InvertedBottleneckBlock(
        nn.Conv2d,
        input_channels=2,
        output_channels=2,
        kernel_size=3,
        stride=1,
        expansion_ratio=1,
        drop_rate=0.5,
    )
    block.expand = nn.Identity()
    block.depthwise = nn.Identity()
    block.project = nn.Module()
    block.project.conv = nn.Identity()
    x = torch.ones(32, 2, 2, 2)

    block.train()
    torch.manual_seed(0)
    output = block(x)
    assert set(output[:, 0, 0, 0].tolist()) == {1.0, 3.0}
    block.eval()
    torch.testing.assert_close(block(x), 2 * x)


@pytest.mark.parametrize("drop_rate", [-0.1, 1.0, float("inf"), True])
def test_stochastic_depth_rejects_invalid_rates(drop_rate):
    with pytest.raises(ValueError, match="drop_rate"):
        _small_model(drop_rate=drop_rate)


def test_removed_se_and_tiling_apis_are_rejected():
    with pytest.raises(TypeError, match="se"):
        _small_model(se={})
    with pytest.raises(TypeError, match="grid_size"):
        _small_model(cc={"grid_size": None})


def test_public_config_and_model_signatures():
    assert set(CCConfig.__dataclass_fields__) == {
        "enabled",
        "num_experts",
        "reduction",
        "use_std",
        "expert_dropout",
        "num_groups",
    }
    defaults = CCConfig()
    assert defaults.reduction is None
    assert defaults.use_std is False
    assert defaults.expert_dropout == 0
    assert defaults.num_groups == 1
    model_parameters = inspect.signature(CondUNet).parameters
    assert "se" not in model_parameters
    decoder_parameters = inspect.signature(CondUNetDecoder).parameters
    assert not {"num_experts", "cc", "reduction", "use_std"} & set(decoder_parameters)


@pytest.mark.parametrize(
    "config",
    [
        {"reduction": 0.5},
        {"reduction": float("inf")},
        {"use_std": 1},
        {"num_groups": 0},
        {"num_groups": 1.5},
    ],
)
def test_cc_settings_are_validated(config):
    with pytest.raises(ValueError):
        _small_model(cc=config)


def test_cc_requires_experts_when_enabled():
    with pytest.raises(ValueError, match="num_experts is 0"):
        _small_model(cc={"enabled": True})


def _trainer_for_schedule(model, anneal_epochs):
    trainer = object.__new__(nnUNetTrainerAdamW)
    trainer.network = model
    trainer.expert_dropout_anneal_epochs = anneal_epochs
    trainer.current_epoch = 0
    return trainer


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [(0, 0.6), (2, 0.3), (4, 0.0), (8, 0.0)],
)
def test_adamw_expert_dropout_linear_schedule(epoch, expected):
    model = _small_model(
        cc={
            "enabled": [[True], [False, False], [False]],
            "num_experts": 2,
            "expert_dropout": 0.6,
        }
    )
    trainer = _trainer_for_schedule(model, 4)
    trainer.current_epoch = epoch
    trainer._update_expert_dropout()
    router = model.encoder.stages[0].blocks[0].router
    assert router.active_expert_dropout == pytest.approx(expected)


def test_adamw_expert_dropout_schedule_none_leaves_rate_unchanged():
    model = _small_model(
        cc={
            "enabled": [[True], [False, False], [False]],
            "num_experts": 2,
            "expert_dropout": 0.4,
        }
    )
    trainer = _trainer_for_schedule(model, None)
    trainer.current_epoch = 100
    trainer._update_expert_dropout()
    assert model.encoder.stages[0].blocks[0].router.active_expert_dropout == pytest.approx(0.4)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_adamw_expert_dropout_anneal_epochs_validation(value):
    trainer = object.__new__(nnUNetTrainerAdamW)
    trainer.configuration_manager = SimpleNamespace(
        trainer={"expert_dropout_anneal_epochs": value}
    )
    trainer.initial_lr = 3e-4
    trainer.weight_decay = 1e-3
    trainer.num_epochs = 250
    trainer.warmup_epochs = 5
    trainer.min_lr = 1e-6
    trainer.enable_deep_supervision = False
    trainer.expert_dropout_anneal_epochs = None
    with pytest.raises((TypeError, ValueError), match="expert_dropout_anneal_epochs"):
        trainer._apply_trainer_configuration()
