
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr
from scipy.ndimage import gaussian_filter
import sys

sys.path.insert(0, os.path.dirname(__file__))
from models.vae import VAEEncoder, VAEDecoder
from models.vre import VRE

# ── Constants ────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP   = 64
STEP   = 48

SEASON_MAP = {
    "Monsoon (Jun–Sep)": 0,
    "Winter (Dec–Feb)": 1,
    "Summer (Mar–May)": 2,
    "Post-Monsoon (Oct–Nov)": 3
}

VAR_NAMES = [
    "2m Temp", "Dewpoint", "Rel Humidity", "Wind Speed", "Precipitation",
    "Var6", "Var7", "Var8", "Var9", "Var10",
    "Var11", "Var12", "Var13", "Var14", "Var15",
    "Var16", "Var17", "Var18", "Var19", "Var20",
    "Var21", "Var22", "Var23", "Var24", "Var25",
    "Var26", "Var27", "Var28"
]

# ── Load model ───────────────────────────────────────────────
def load_model():
    ckpt_path = "phyda_india_best.pth"
    vae_enc = VAEEncoder(in_channels=28, latent_channels=32).to(DEVICE)
    vae_dec = VAEDecoder(latent_channels=32, out_channels=28).to(DEVICE)
    vre     = VRE(in_channels=28, latent_channels=32).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    vae_enc.load_state_dict(ckpt["vae_enc"])
    vae_dec.load_state_dict(ckpt["vae_dec"])
    vre.load_state_dict(ckpt["vre"])

    vae_enc.eval(); vae_dec.eval(); vre.eval()
    return vae_enc, vae_dec, vre

vae_enc, vae_dec, vre = load_model()

mean_np = np.load("norm_mean.npy")
std_np  = np.load("norm_std_fixed.npy")
mean_t  = torch.tensor(mean_np, dtype=torch.float32).to(DEVICE)
std_t   = torch.tensor(std_np,  dtype=torch.float32).to(DEVICE)

def normalize(x):
    return (x - mean_t) / (std_t + 1e-6)

def denormalize(x):
    return x * (std_t + 1e-6) + mean_t

# ── Tiled inference ──────────────────────────────────────────
def tiled_inference(Xb_t, Xobs_t, season_idx):
    B, C, H, W = Xb_t.shape
    out  = torch.zeros_like(Xb_t)
    wmap = torch.zeros(B, 1, H, W, device=DEVICE)

    for hi in range(0, H - CROP + 1, STEP):
        for wi in range(0, W - CROP + 1, STEP):
            xb_c   = Xb_t[:, :, hi:hi+CROP, wi:wi+CROP]
            xobs_c = Xobs_t[:, :, hi:hi+CROP, wi:wi+CROP]

            xb_n   = normalize(xb_c)
            xobs_n = normalize(xobs_c)

            with torch.no_grad():
                _, mu_b, _  = vae_enc(xb_n)
                _, mu_o, _  = vae_enc(xobs_n)
                trust       = vre(xobs_n, season_idx)

            obs_mask = (xobs_c.abs().sum(dim=1, keepdim=True) > 0).float()
            Xa_n     = mu_b + trust * obs_mask * (mu_o - mu_b)
            Xa_c     = denormalize(Xa_n)

            from scipy.ndimage import gaussian_filter
            w = np.ones((CROP, CROP))
            w = torch.tensor(gaussian_filter(w, sigma=8),
                             dtype=torch.float32, device=DEVICE)

            out[:, :, hi:hi+CROP, wi:wi+CROP]  += Xa_c * w
            wmap[:, :, hi:hi+CROP, wi:wi+CROP] += w

    Xa = out / (wmap + 1e-6)
    return Xa

# ── Skill score ───────────────────────────────────────────────
def compute_skill(Xb, Xa, Xobs):
    mask  = (Xobs.abs().sum(axis=1, keepdims=True) > 0)
    err_b = np.where(mask, (Xb - Xobs)**2, 0).sum()
    err_a = np.where(mask, (Xa - Xobs)**2, 0).sum()
    denom = np.where(mask, Xobs**2, 0).sum() + 1e-6
    skill = 1.0 - (err_a / (err_b + 1e-6))
    return float(np.clip(skill * 100, 0, 100))

# ── Plot ──────────────────────────────────────────────────────
def make_plot(Xb, Xa, Xobs, var_idx, var_name, season_name, skill):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#0e1117")

    titles = ["Background (Xb)", "Analysis (Xa)", "Observations (Xobs)"]
    arrays = [Xb[0, var_idx], Xa[0, var_idx], Xobs[0, var_idx]]
    cmaps  = ["coolwarm", "RdYlBu_r", "viridis"]

    vmin = min(a.min() for a in arrays)
    vmax = max(a.max() for a in arrays)

    for ax, arr, title, cmap in zip(axes, arrays, titles, cmaps):
        im = ax.contourf(arr, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, color="white", fontsize=11)
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"PhyDA-India | {var_name} | {season_name} | Skill: {skill:.2f}%",
        color="white", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    return fig

# ── Main inference function ───────────────────────────────────
def run_inference(xb_file, xobs_file, season_name, var_choice):
    try:
        Xb   = np.load(xb_file.name).astype(np.float32)
        Xobs = np.load(xobs_file.name).astype(np.float32)

        if Xb.ndim == 3:
            Xb   = Xb[np.newaxis]
        if Xobs.ndim == 3:
            Xobs = Xobs[np.newaxis]

        season_idx = torch.tensor(
            [SEASON_MAP[season_name]], dtype=torch.long, device=DEVICE
        )

        Xb_t   = torch.tensor(Xb,   dtype=torch.float32).to(DEVICE)
        Xobs_t = torch.tensor(Xobs, dtype=torch.float32).to(DEVICE)

        Xa_t = tiled_inference(Xb_t, Xobs_t, season_idx)
        Xa   = Xa_t.cpu().numpy()

        skill   = compute_skill(Xb, Xa, Xobs)
        var_idx = VAR_NAMES.index(var_choice)
        fig     = make_plot(Xb, Xa, Xobs, var_idx, var_choice, season_name, skill)

        summary = (
            f"✅ Assimilation complete\n"
            f"📊 Skill Score : {skill:.2f}%\n"
            f"🌦️  Season      : {season_name}\n"
            f"📌 Variable    : {var_choice}\n"
            f"📐 Grid shape  : {Xb.shape}"
        )
        return fig, summary

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Base(), title="PhyDA-India") as demo:
    gr.Markdown("""
    # 🌧️ PhyDA-India
    ### Physics-guided Deep Learning Data Assimilation for Indian Weather Prediction
    Upload your background (`Xb`) and observation (`Xobs`) `.npy` files, select the season and variable to visualize.
    """)

    with gr.Row():
        xb_file   = gr.File(label="Upload Xb file (.npy)", file_types=[".npy"])
        xobs_file = gr.File(label="Upload Xobs file (.npy)", file_types=[".npy"])

    with gr.Row():
        season_dd = gr.Dropdown(
            choices=list(SEASON_MAP.keys()),
            value="Monsoon (Jun–Sep)",
            label="Season"
        )
        var_dd = gr.Dropdown(
            choices=VAR_NAMES,
            value="2m Temp",
            label="Variable to visualize"
        )

    run_btn = gr.Button("▶ Run Assimilation", variant="primary")

    with gr.Row():
        plot_out    = gr.Plot(label="Assimilation Output")
        summary_out = gr.Textbox(label="Summary", lines=6)

    run_btn.click(
        fn=run_inference,
        inputs=[xb_file, xobs_file, season_dd, var_dd],
        outputs=[plot_out, summary_out]
    )

    gr.Markdown("""
    ---
    **Authors:** Dharshini G, Gopika D | Chennai Institute of Technology  
    **Supervisor:** Dr. Venkatesan, NIT Puducherry  
    **Based on:** [PhyDA (arXiv:2505.12882)](https://arxiv.org/abs/2505.12882)
    """)

demo.launch()
