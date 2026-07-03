import unittest
import torch
from nnunetv2.training.network_architecture.cond_unet import (
    TiledPoolMLP,
    SqueezeAndExcitationBlock,
    Router,
    CondPWConv,
)


class TestCondUNetBlocks(unittest.TestCase):
    def test_tiled_pool_mlp_validation(self):
        # reduction <= 0 validation
        with self.assertRaises(ValueError):
            TiledPoolMLP(input_channels=16, output_channels=8, reduction=0, nonlin=torch.nn.ReLU)
        with self.assertRaises(ValueError):
            TiledPoolMLP(input_channels=16, output_channels=8, reduction=-0.5, nonlin=torch.nn.ReLU)
        # tile_size values <= 0 validation
        with self.assertRaises(ValueError):
            TiledPoolMLP(input_channels=16, output_channels=8, reduction=0.5, nonlin=torch.nn.ReLU, tile_size=[0, 4])

    def test_tiled_pool_mlp_forward_global(self):
        # 2D global average pooling (no tiling)
        module = TiledPoolMLP(input_channels=8, output_channels=4, reduction=0.5, nonlin=torch.nn.ReLU)
        x = torch.randn(2, 8, 16, 16)
        out = module(x)
        self.assertEqual(out.shape, (2, 4))

    def test_tiled_pool_mlp_forward_tiled_2d(self):
        # 2D tiled average pooling
        module = TiledPoolMLP(
            input_channels=8, output_channels=4, reduction=0.5, nonlin=torch.nn.ReLU, tile_size=(4, 4)
        )
        x = torch.randn(2, 8, 16, 16)
        out = module(x)
        # x.shape[2:] / tile_size => 16 // 4 = 4
        # shape => [B, *grid_shape, output_channels] = [2, 4, 4, 4]
        self.assertEqual(out.shape, (2, 4, 4, 4))

        # Check dimension mismatch
        with self.assertRaises(ValueError):
            module(torch.randn(2, 8, 16)) # 1D spatial input, expected 2D spatial dims

    def test_se_block_forward(self):
        # Global mode
        se_global = SqueezeAndExcitationBlock(channels=8, reduction=0.5, nonlin=torch.nn.ReLU)
        x = torch.randn(2, 8, 16, 16)
        out = se_global(x)
        self.assertEqual(out.shape, x.shape)

        # Tiled mode
        se_tiled = SqueezeAndExcitationBlock(channels=8, reduction=0.5, nonlin=torch.nn.ReLU, tile_size=(4, 4))
        out_tiled = se_tiled(x)
        self.assertEqual(out_tiled.shape, x.shape)

    def test_router_forward(self):
        # Global mode
        router_global = Router(input_channels=8, num_experts=3, reduction=0.5, nonlin=torch.nn.ReLU)
        x = torch.randn(2, 8, 16, 16)
        out = router_global(x)
        self.assertEqual(out.shape, (2, 3))

        # Tiled mode
        router_tiled = Router(input_channels=8, num_experts=3, reduction=0.5, nonlin=torch.nn.ReLU, tile_size=(4, 4))
        out_tiled = router_tiled(x)
        self.assertEqual(out_tiled.shape, (2, 4, 4, 3))

    def test_cond_pw_conv_forward_global(self):
        # 1. num_groups = 1, global routing
        conv_op = torch.nn.Conv2d(8, 16, kernel_size=1)
        layer = CondPWConv(
            conv=conv_op,
            num_experts=3,
            router_reduction=0.5,
            nonlin=torch.nn.ReLU,
            use_internal_router=True,
            num_groups=1,
        )
        x = torch.randn(2, 8, 16, 16)
        out = layer(x)
        self.assertEqual(out.shape, (2, 16, 16, 16))

        # 2. num_groups > 1, group_on_out = True
        layer_g_out = CondPWConv(
            conv=conv_op,
            num_experts=3,
            router_reduction=0.5,
            nonlin=torch.nn.ReLU,
            use_internal_router=True,
            num_groups=2,
            group_on_out=True,
        )
        out_g_out = layer_g_out(x)
        self.assertEqual(out_g_out.shape, (2, 16, 16, 16))

        # 3. num_groups > 1, group_on_out = False
        layer_g_in = CondPWConv(
            conv=conv_op,
            num_experts=3,
            router_reduction=0.5,
            nonlin=torch.nn.ReLU,
            use_internal_router=True,
            num_groups=2,
            group_on_out=False,
        )
        out_g_in = layer_g_in(x)
        self.assertEqual(out_g_in.shape, (2, 16, 16, 16))

    def test_cond_pw_conv_forward_tiled(self):
        # Tiled routing
        conv_op = torch.nn.Conv2d(8, 16, kernel_size=1)
        layer = CondPWConv(
            conv=conv_op,
            num_experts=3,
            router_reduction=0.5,
            nonlin=torch.nn.ReLU,
            use_internal_router=False, # pass external router scores
            num_groups=2,
            group_on_out=True,
        )
        scores = torch.randn(2, 4, 4, 3 * 2) # [B, grid_h, grid_w, num_experts * num_groups]
        x = torch.randn(2, 8, 16, 16)
        out = layer(x, scores=scores)
        self.assertEqual(out.shape, (2, 16, 16, 16))

        # Test _blend_experts output shape
        flat_scores = scores.reshape(-1, 3 * 2)
        weight, bias = layer._blend_experts(flat_scores)
        # N = 2 * 4 * 4 = 32
        self.assertEqual(weight.shape, (32, 16, 8))
        self.assertEqual(bias.shape, (32, 16))
