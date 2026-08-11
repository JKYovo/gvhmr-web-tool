import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

from hmr4d.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock
from hmr4d.network.base_arch.transformer.layer import zero_module
from hmr4d.utils.net_utils import length_to_mask


class FootEncoderRoPE(nn.Module):
    """FootMR temporal refiner predicting residual global ankle rotations."""

    def __init__(
        self,
        max_len=120,
        cliffcam_dim=3,
        latent_dim=256,
        num_layers=6,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
        attention_impl="dense",
        attention_chunk_size=128,
    ):
        super().__init__()
        self.num_2d_joints = 8
        self.num_rot_condition = 4
        self.output_dim = 12
        self.max_len = max_len
        self.latent_dim = latent_dim
        self.attention_impl = attention_impl

        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(self.num_2d_joints, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            self.num_2d_joints * 32,
            hidden_features=latent_dim * 2,
            out_features=latent_dim,
            drop=dropout,
        )
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        self.rot6d_embedder = nn.Sequential(
            nn.Linear(6 * self.num_rot_condition, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        self.blocks = nn.ModuleList(
            [
                EncoderRoPEBlock(
                    latent_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_impl=attention_impl,
                    attention_chunk_size=attention_chunk_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = Mlp(latent_dim, out_features=self.output_dim)

    def forward(self, length, obs, f_cliffcam, global_rot6d):
        B, L, J, C = obs.shape
        assert J == self.num_2d_joints and C == 3

        obs = obs.clone()
        visible_mask = obs[..., [2]] > 0.5
        obs[~visible_mask[..., 0]] = 0
        f_obs = self.learned_pos_linear(obs[..., :2])
        missing_tokens = self.learned_pos_params.repeat(B, L, 1, 1)
        f_obs = f_obs * visible_mask + missing_tokens * ~visible_mask
        x = self.embed_noisyobs(f_obs.view(B, L, -1))
        x = x + self.cliffcam_embedder(f_cliffcam) + self.rot6d_embedder(global_rot6d)

        padding_mask = ~length_to_mask(length, L)
        if L > self.max_len:
            if self.attention_impl == "local":
                attention_mask = ("local", self.max_len)
            else:
                attention_mask = torch.ones((L, L), device=x.device, dtype=torch.bool)
                for i in range(L):
                    min_ind = max(0, i - self.max_len // 2)
                    max_ind = min(L, i + self.max_len // 2)
                    max_ind = max(self.max_len, max_ind)
                    min_ind = min(L - self.max_len, min_ind)
                    attention_mask[i, min_ind:max_ind] = False
        else:
            attention_mask = None

        for block in self.blocks:
            x = block(x, attn_mask=attention_mask, tgt_key_padding_mask=padding_mask)

        # The network predicts a residual around the initial global ankles.
        return self.final_layer(x) + global_rot6d[:, :, -12:]
