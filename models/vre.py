
import torch
import torch.nn as nn

class VRE(nn.Module):
    def __init__(self, in_channels=28, latent_channels=32, num_seasons=4):
        super().__init__()

        # Multi-scale CNN feature extractor
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU()
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2), nn.ReLU()
        )
        self.cnn7 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, padding=3), nn.ReLU()
        )

        # Fuse 3 scales → 64 channels
        self.fuse = nn.Sequential(
            nn.Conv2d(96, 64, 1), nn.ReLU()
        )

        # Season embedding
        self.season_emb = nn.Embedding(num_seasons, 64)

        # One attention head per season
        self.attn_heads = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
            for _ in range(num_seasons)
        ])

        # Gate: season embedding → scalar gate per token
        self.gate_fc = nn.Linear(64, 1)

        # Project to latent channels
        self.proj = nn.Conv2d(64, latent_channels, 1)

        # Trust head: latent → scalar in (0,1)
        self.trust_fc = nn.Linear(latent_channels, 1)

    def forward(self, x_obs, season_idx):
        """
        x_obs      : (B, 28, H, W)
        season_idx : (B,)  int  0=Monsoon 1=Winter 2=Summer 3=Post-monsoon
        returns    : trust (B, 1, 1, 1)  in range [0.3, 0.7]
        """
        B, C, H, W = x_obs.shape

        # Multi-scale features
        f3 = self.cnn3(x_obs)
        f5 = self.cnn5(x_obs)
        f7 = self.cnn7(x_obs)
        f  = self.fuse(torch.cat([f3, f5, f7], dim=1))  # (B, 64, H, W)

        # Flatten spatial → tokens
        tokens = f.flatten(2).permute(0, 2, 1)           # (B, H*W, 64)

        # Season-aware attention (use head for majority season in batch)
        s         = int(season_idx[0].item())
        se        = self.season_emb(season_idx)           # (B, 64)
        gate      = torch.sigmoid(self.gate_fc(se))       # (B, 1)
        attn_out, _ = self.attn_heads[s](tokens, tokens, tokens)
        tokens    = tokens + gate.unsqueeze(1) * attn_out # (B, H*W, 64)

        # Reshape back to spatial
        feat = tokens.permute(0, 2, 1).view(B, 64, H, W) # (B, 64, H, W)

        # Project to latent
        z = self.proj(feat)                               # (B, 32, H, W)

        # Trust scalar
        z_gap      = z.mean(dim=[2, 3])                   # (B, 32)
        trust_raw  = self.trust_fc(z_gap)                 # (B, 1)
        trust      = 0.3 + 0.4 * torch.sigmoid(trust_raw)# (B, 1)  → [0.3, 0.7]
        trust_4d   = trust.view(B, 1, 1, 1)

        return trust_4d
