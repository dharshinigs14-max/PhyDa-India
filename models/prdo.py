
import torch
import torch.nn as nn

# Channel indices in the 28-channel ERA5 stack
# Adjust these if your channel order differs
CH_TEMP  = 0   # 2m temperature
CH_DEWP  = 1   # dewpoint temperature
CH_HUM   = 2   # relative humidity
CH_WIND  = 3   # wind speed
CH_RAIN  = 4   # precipitation

def clausius_clapeyron_loss(x_pred, season_mask):
    """
    Monsoon physics: warmer air must hold more moisture.
    Penalise cases where humidity drops as temperature rises.
    """
    if season_mask.sum() == 0:
        return torch.tensor(0.0, device=x_pred.device)
    xm   = x_pred[season_mask]
    temp = xm[:, CH_TEMP]
    hum  = xm[:, CH_HUM]
    # dT and dQ should have same sign in monsoon
    dT   = temp - temp.mean(dim=[1,2], keepdim=True)
    dQ   = hum  - hum.mean(dim=[1,2],  keepdim=True)
    violation = torch.relu(-(dT * dQ))   # penalise anti-correlation
    return violation.mean()

def bowen_ratio_loss(x_pred, season_mask):
    """
    Sensible / latent heat ratio constraint.
    In monsoon: high humidity → Bowen ratio < 1 (latent > sensible).
    Penalise when temp anomaly >> humidity anomaly.
    """
    if season_mask.sum() == 0:
        return torch.tensor(0.0, device=x_pred.device)
    xm   = x_pred[season_mask]
    temp = xm[:, CH_TEMP]
    hum  = xm[:, CH_HUM]
    bowen = (temp.std() + 1e-6) / (hum.std() + 1e-6)
    return torch.relu(bowen - 2.0)   # allow up to ratio 2

def thermal_inertia_loss(x_pred, season_mask):
    """
    Spatial smoothness of temperature field.
    Large abrupt gradients are physically unrealistic.
    """
    if season_mask.sum() == 0:
        return torch.tensor(0.0, device=x_pred.device)
    xm   = x_pred[season_mask]
    temp = xm[:, CH_TEMP]
    dx   = (temp[:, :, 1:] - temp[:, :, :-1]).abs().mean()
    dy   = (temp[:, 1:, :] - temp[:, :-1, :]).abs().mean()
    return dx + dy

def monsoon_physics_loss(x_pred, season_idx):
    """
    Combined physics loss — only active for monsoon batches.
    season_idx: (B,)  0=Monsoon 1=Winter 2=Summer 3=Post-monsoon
    """
    monsoon_mask = (season_idx == 0)

    cc   = clausius_clapeyron_loss(x_pred, monsoon_mask)
    br   = bowen_ratio_loss(x_pred, monsoon_mask)
    ti   = thermal_inertia_loss(x_pred, monsoon_mask)

    return cc + 0.5 * br + 0.1 * ti
