import unittest
import torch
from nnunetv2.training.network_architecture.cond_unet import (
    TiledPoolMLP,
    SqueezeAndExcitationBlock,
    Router,
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
