
import numpy as np
import torch

class Normalizer:
    def __init__(self, data_dir):
        mean = np.load(f"{data_dir}/norm_mean.npy")
        std  = np.load(f"{data_dir}/norm_std_fixed.npy")
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std  = torch.tensor(std,  dtype=torch.float32)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)
        return self

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-6)

    def denormalize(self, x):
        return x * (self.std + 1e-6) + self.mean
