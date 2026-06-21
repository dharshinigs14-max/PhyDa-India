
import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    def __init__(self, in_channels=28, latent_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),  nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),           nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),           nn.ReLU(),
        )
        self.mu_head      = nn.Conv2d(64, latent_channels, 1)
        self.log_var_head = nn.Conv2d(64, latent_channels, 1)

    def forward(self, x):
        h = self.encoder(x)
        mu      = self.mu_head(h)
        log_var = self.log_var_head(h)
        std     = torch.exp(0.5 * log_var)
        eps     = torch.randn_like(std)
        z       = mu + eps * std
        return z, mu, log_var


class VAEDecoder(nn.Module):
    def __init__(self, latent_channels=32, out_channels=28):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64,  3, padding=1), nn.ReLU(),
            nn.Conv2d(64,             128,  3, padding=1), nn.ReLU(),
            nn.Conv2d(128,             64,  3, padding=1), nn.ReLU(),
            nn.Conv2d(64,  out_channels,    1),
        )

    def forward(self, z):
        return self.decoder(z)
