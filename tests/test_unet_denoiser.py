import unittest

import torch

from models.diffusion import UNetDenoiser


class UNetDenoiserShapeTest(unittest.TestCase):
    def test_unet_denoiser_output_shape(self) -> None:
        torch.manual_seed(0)
        model = UNetDenoiser(
            target_dim=4,
            condition_dim=12,
            in_H=8,
            in_W=8,
            down_out_dims=[4, 8],
            mid_out_dims=[8, 8],
            up_out_dims=[8, 4],
            down_transformer_model_dims=[8, 8],
            mid_transformer_model_dims=[8, 8],
            up_transformer_model_dims=[8, 8],
            transformer_feedforward_dim=16,
            n_conv_layers_per_scaling_block=2,
            n_transformer_encoder_layers_per_scaling_block=1,
            n_transformer_decoder_layers_per_scaling_block=1,
            n_conv_layers_per_mid_block=2,
            n_transformer_encoder_layers_per_mid_block=1,
            n_transformer_decoder_layers_per_mid_block=1,
            n_attention_heads=1,
            transformer_maxlength=4,
            switch_ratio=0.0,
        )
        model.eval()

        batch_size = 2
        target = torch.randn(batch_size, 2, 4, 8, 8)
        condition_mu = torch.randn(batch_size, 4, 12)
        condition_logvar = torch.randn(batch_size, 4, 12)
        step = torch.zeros(batch_size, 1, dtype=torch.long)
        days = torch.arange(4, dtype=torch.long).repeat(batch_size, 1)

        output = model(
            target=target,
            condition_mu=condition_mu,
            condition_logvar=condition_logvar,
            integer_step=step,
            condition_days=days,
        )

        self.assertEqual(output.shape, target.shape)


if __name__ == "__main__":
    unittest.main()
