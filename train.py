
import os, gc, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/content/drive/MyDrive/PhyDA")
from models.vae       import VAEEncoder, VAEDecoder
from models.vre       import VRE
from models.prdo      import monsoon_physics_loss
from utils.normalizer import Normalizer

DATA_DIR   = "/content/drive/MyDrive/PhyDA/data"
RUNS_DIR   = "/content/drive/MyDrive/PhyDA/runs"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 20
BATCH_SIZE = 4
CROP       = 64
LR         = 1e-4

MONTH_SEASON = {
    1:1, 2:1, 3:2, 4:2, 5:2,
    6:0, 7:0, 8:0, 9:0,
    10:3, 11:3, 12:1
}

TRAIN_YEARS = [2022, 2023]
VAL_YEARS   = [2024]
ALL_MONTHS  = list(range(1, 13))

class CropDataset(Dataset):
    def __init__(self, Xb, Xobs, n_crops=200):
        self.Xb   = torch.tensor(Xb,   dtype=torch.float32)
        self.Xobs = torch.tensor(Xobs, dtype=torch.float32)
        self.T    = Xb.shape[0]
        self.n    = n_crops
        self.H    = Xb.shape[2]
        self.W    = Xb.shape[3]

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        t  = np.random.randint(0, self.T)
        hi = np.random.randint(0, self.H - CROP + 1)
        wi = np.random.randint(0, self.W - CROP + 1)
        xb   = self.Xb  [t, :, hi:hi+CROP, wi:wi+CROP]
        xobs = self.Xobs[t, :, hi:hi+CROP, wi:wi+CROP]
        return xb, xobs

def kl_loss(mu, log_var):
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

def obs_supervised_loss(x_pred, Xobs):
    mask  = (Xobs.abs().sum(dim=1, keepdim=True) > 0).float()
    diff  = (x_pred - Xobs) * mask
    denom = mask.sum() + 1e-6
    return (diff ** 2).sum() / denom

def run_epoch(year, month, models, optimizers, norm, is_train):
    vae_enc, vae_dec, vre = models
    n_crops = 200 if is_train else 100

    Xb   = np.load(f"{DATA_DIR}/Xb_{year}_{month:02d}.npy").astype(np.float32)
    Xobs = np.load(f"{DATA_DIR}/Xobs_{year}_{month:02d}.npy").astype(np.float32)

    ds     = CropDataset(Xb, Xobs, n_crops=n_crops)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=is_train,
                        num_workers=0, pin_memory=False)

    season     = MONTH_SEASON[month]
    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, xobs in loader:
            xb    = norm.normalize(xb.to(DEVICE))
            xobs  = norm.normalize(xobs.to(DEVICE))
            s_idx = torch.full((xb.shape[0],), season,
                               dtype=torch.long, device=DEVICE)

            z_bg,  mu_b, lv_b = vae_enc(xb)
            z_obs, mu_o, _    = vae_enc(xobs)
            trust             = vre(xobs, s_idx)
            z_analysis        = (1 - trust) * mu_b + trust * mu_o
            x_pred            = vae_dec(z_analysis)

            recon  = nn.functional.mse_loss(x_pred, xb)
            obs_l  = obs_supervised_loss(x_pred, xobs)
            kl     = kl_loss(mu_b, lv_b)
            phys   = monsoon_physics_loss(x_pred, s_idx)
            loss   = recon + 5.0*obs_l + 0.001*kl + 1.0*phys

            if is_train:
                for opt in optimizers:
                    opt.zero_grad()
                loss.backward()
                for opt in optimizers:
                    opt.step()

            total_loss += loss.item()
            n_batches  += 1

    del Xb, Xobs, ds, loader
    gc.collect()
    torch.cuda.empty_cache()

    return total_loss / max(n_batches, 1)

def train():
    os.makedirs(RUNS_DIR, exist_ok=True)
    norm    = Normalizer(DATA_DIR).to(DEVICE)
    vae_enc = VAEEncoder(in_channels=28, latent_channels=32).to(DEVICE)
    vae_dec = VAEDecoder(latent_channels=32, out_channels=28).to(DEVICE)
    vre     = VRE(in_channels=28, latent_channels=32).to(DEVICE)

    models     = (vae_enc, vae_dec, vre)
    optimizers = (
        torch.optim.Adam(vae_enc.parameters(), lr=LR),
        torch.optim.Adam(vae_dec.parameters(), lr=LR),
        torch.optim.Adam(vre.parameters(),     lr=LR),
    )

    # ── resume if checkpoint exists
    start_epoch = 1
    best_val    = float("inf")
    ckpt_path   = f"{RUNS_DIR}/phyda_india_best.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        vae_enc.load_state_dict(ckpt["vae_enc"])
        vae_dec.load_state_dict(ckpt["vae_dec"])
        vre.load_state_dict(ckpt["vre"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["val_loss"]
        print(f"▶ Resuming from epoch {start_epoch}, best_val={best_val:.4f}")

    for epoch in range(start_epoch, EPOCHS + 1):
        # train
        vae_enc.train(); vae_dec.train(); vre.train()
        train_loss = 0.0
        for year in TRAIN_YEARS:
            for month in ALL_MONTHS:
                m_loss = run_epoch(year, month, models, optimizers, norm, is_train=True)
                train_loss += m_loss
                print(f"  train {year}-{month:02d}  loss={m_loss:.4f}")

        # val
        vae_enc.eval(); vae_dec.eval(); vre.eval()
        val_loss = 0.0
        for year in VAL_YEARS:
            for month in ALL_MONTHS:
                m_loss = run_epoch(year, month, models, None, norm, is_train=False)
                val_loss += m_loss
                print(f"  val   {year}-{month:02d}  loss={m_loss:.4f}")

        train_loss /= (len(TRAIN_YEARS) * 12)
        val_loss   /= (len(VAL_YEARS)   * 12)

        print(f"\nEpoch {epoch:02d}/{EPOCHS} "
              f"train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch"   : epoch,
                "vae_enc" : vae_enc.state_dict(),
                "vae_dec" : vae_dec.state_dict(),
                "vre"     : vre.state_dict(),
                "val_loss": best_val,
            }, ckpt_path)
            print(f"  ✅ saved best checkpoint  val={best_val:.4f}\n")

    print("\n Training complete!")
    print(f"   Best val loss : {best_val:.4f}")
    print(f"   Checkpoint    : {ckpt_path}")
