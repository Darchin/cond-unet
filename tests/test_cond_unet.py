"""Tests for the refactored CondUNet architecture."""
import pytest
import torch
import torch.nn as nn
import numpy as np

from nnunetv2.training.network_architecture.cond_unet import (
    SEConfig,
    CCConfig,
    StemConfig,
    _normalize_config,
    _expand_int_param,
    _expand_expansion_ratios,
    _expand_block_config,
    TiledPoolMLP,
    SqueezeAndExcitationBlock,
    Router,
    CondPWConv,
    DepthwiseConvBlock,
    InvertedBottleneckBlock,
    StackedCondInvertedBottleneckBlocks,
    CondUNetEncoder,
    CondUNetDecoder,
    CondUNet,
)


# --- Config dataclass tests ---

class TestSEConfig:
    def test_defaults(self):
        cfg = SEConfig()
        assert cfg.reduction == 0.125
        assert cfg.encoder is False
        assert cfg.decoder is False
        assert cfg.tile_size is None

    def test_from_dict(self):
        cfg = SEConfig.from_dict({"reduction": 0.25, "encoder": True})
        assert cfg.reduction == 0.25
        assert cfg.encoder is True
        assert cfg.decoder is False

    def test_round_trip(self):
        cfg = SEConfig(reduction=0.5, encoder=[True, False], tile_size=[16, 16])
        d = cfg.to_dict()
        cfg2 = SEConfig.from_dict(d)
        assert cfg == cfg2


class TestCCConfig:
    def test_defaults(self):
        cfg = CCConfig()
        assert cfg.encoder_num_experts == 0
        assert cfg.encoder_num_groups == 1

    def test_from_dict(self):
        cfg = CCConfig.from_dict({"encoder": True, "encoder_num_experts": 4})
        assert cfg.encoder is True
        assert cfg.encoder_num_experts == 4


class TestStemConfig:
    def test_defaults(self):
        cfg = StemConfig()
        assert cfg.channels is None
        assert cfg.kernel_size == 3
        assert cfg.stride == 1


class TestNormalizeConfig:
    def test_none_gives_defaults(self):
        cfg = _normalize_config(None, SEConfig)
        assert cfg == SEConfig()

    def test_dict_converts(self):
        cfg = _normalize_config({"reduction": 0.5}, SEConfig)
        assert isinstance(cfg, SEConfig)
        assert cfg.reduction == 0.5

    def test_instance_passthrough(self):
        original = SEConfig(reduction=0.3)
        cfg = _normalize_config(original, SEConfig)
        assert cfg is original

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _normalize_config(42, SEConfig)


# --- Helper function tests ---

class TestExpandIntParam:
    def test_scalar_expansion(self):
        assert _expand_int_param(3, 4, "test") == [3, 3, 3, 3]

    def test_list_passthrough(self):
        assert _expand_int_param([1, 2, 3], 3, "test") == [1, 2, 3]

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            _expand_int_param([1, 2], 3, "test")

    def test_min_value_enforced(self):
        with pytest.raises(ValueError):
            _expand_int_param(0, 2, "test", min_value=1)

    def test_min_value_zero_allows_zero(self):
        assert _expand_int_param(0, 2, "test", min_value=0) == [0, 0]


# --- Module tests ---

class TestTiledPoolMLP:
    def test_global_output_shape(self):
        mlp = TiledPoolMLP(16, 8, 0.5, nn.ReLU)
        x = torch.randn(2, 16, 10, 10)
        out = mlp(x)
        assert out.shape == (2, 8)

    def test_tiled_output_shape(self):
        mlp = TiledPoolMLP(16, 8, 0.5, nn.ReLU, tile_size=(5, 5))
        x = torch.randn(2, 16, 10, 10)
        out = mlp(x)
        assert out.shape == (2, 2, 2, 8)  # grid 10//5=2 per dim


class TestDepthwiseConvBlock:
    def test_output_shape(self):
        block = DepthwiseConvBlock(
            nn.Conv2d, 32, [3, 3], [2, 2],
            norm_op=nn.BatchNorm2d, nonlin=nn.ReLU,
        )
        x = torch.randn(1, 32, 16, 16)
        out = block(x)
        assert out.shape == (1, 32, 8, 8)

    def test_depthwise_groups(self):
        block = DepthwiseConvBlock(nn.Conv2d, 32, [3, 3], [1, 1])
        assert block.conv.groups == 32


# --- Integration tests ---

def _make_base_model(**kwargs):
    """Create a minimal 2D CondUNet for testing."""
    defaults = dict(
        input_channels=1,
        n_stages=3,
        features_per_stage=[16, 32, 64],
        conv_op=nn.Conv2d,
        kernel_sizes=3,
        strides=[[1, 1], [2, 2], [2, 2]],
        encoder_n_blocks_per_stage=1,
        num_classes=2,
        decoder_n_blocks_per_stage=1,
        norm_op=nn.InstanceNorm2d,
        nonlin=nn.LeakyReLU,
    )
    defaults.update(kwargs)
    return CondUNet(**defaults)


class TestCondUNetIntegration:
    def test_base_model_forward(self):
        model = _make_base_model()
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        assert out.shape == (1, 2, 32, 32)

    def test_se_from_dict(self):
        model = _make_base_model(se={"encoder": True, "reduction": 0.25})
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        assert out.shape == (1, 2, 32, 32)

    def test_cc_from_dict(self):
        model = _make_base_model(
            cc={"encoder": True, "encoder_num_experts": 4, "reduction": 0.25}
        )
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        assert out.shape == (1, 2, 32, 32)

    def test_tiled_se(self):
        model = _make_base_model(
            se={"encoder": True, "tile_size": [16, 16]}
        )
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        assert out.shape == (1, 2, 32, 32)

    def test_se_and_cc_combined(self):
        model = _make_base_model(
            se={"encoder": True, "decoder": True},
            cc={
                "encoder": True, "decoder": True,
                "encoder_num_experts": 2, "decoder_num_experts": 2,
            },
        )
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        assert out.shape == (1, 2, 32, 32)

    def test_stem_config_dict(self):
        model = _make_base_model(
            stem={"channels": 8, "kernel_size": 5, "stride": 2}
        )
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        assert out.shape == (1, 2, 32, 32)

    def test_transposed_upsampling(self):
        model = _make_base_model(upsample_mode="transposed")
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        assert out.shape == (1, 2, 32, 32)

    def test_compute_conv_feature_map_size(self):
        model = _make_base_model()
        size = model.compute_conv_feature_map_size((32, 32))
        assert isinstance(size, (int, np.integer))
        assert size > 0

    def test_cc_without_experts_raises(self):
        with pytest.raises(ValueError, match="num_experts"):
            _make_base_model(
                cc={"encoder": True, "encoder_num_experts": 0}
            )

    def test_mismatched_tile_sizes_raises(self):
        with pytest.raises(ValueError, match="tile_size"):
            _make_base_model(
                se={"encoder": True, "tile_size": [16, 16]},
                cc={"encoder": True, "encoder_num_experts": 2, "tile_size": [8, 8]},
            )
