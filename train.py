"""
This module includes all the loss and other functions necessary for training the model
Author:mk314k
"""
from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.r3d import R3D


def gan_loss(discriminator_logits:torch.Tensor, is_real:bool)->torch.Tensor:
    """Measure to classify real 3d image with generated 3d image

    Args:
        discriminator_logits (torch.Tensor): raw discriminator output
        is_real (bool): whether the target label is real

    Returns:
        torch.Tensor: mean binary cross entropy
    """
    target = torch.full_like(discriminator_logits, 1.0 if is_real else 0.0)
    return F.binary_cross_entropy_with_logits(discriminator_logits, target)


def vae_loss(recon_logits:torch.Tensor, label:torch.Tensor, pos_weight=1.0)->torch.Tensor:
    """Per-voxel reconstruction loss

    Args:
        recon_logits (torch.Tensor): generated voxel logits (batch, 1, D, H, W)
        label (torch.Tensor): target occupancy in [0, 1] (batch, 1, D, H, W)
        pos_weight (float): weight of occupied voxels; most voxels are empty, so values
            above 1 stop the model from settling on blurry, mostly-empty predictions

    Returns:
        torch.Tensor: mean binary cross entropy over all voxels
    """
    weight = torch.tensor(pos_weight, device=recon_logits.device)
    return F.binary_cross_entropy_with_logits(recon_logits, label, pos_weight=weight)


def voxel_iou(pred:torch.Tensor, label:torch.Tensor, threshold=0.5)->torch.Tensor:
    """
    Args:
        pred (torch.Tensor): occupancy probabilities (batch, ...)
        label (torch.Tensor): target occupancy (batch, ...)
        threshold (float): occupancy cutoff applied to both tensors

    Returns:
        torch.Tensor: intersection over union for each sample (batch,)
    """
    pred = (pred >= threshold).flatten(1)
    label = (label >= threshold).flatten(1)
    intersection = (pred & label).sum(dim=1).float()
    union = (pred | label).sum(dim=1).float()
    return torch.where(union > 0, intersection / union.clamp(min=1), torch.ones_like(union))


def train_step( # pylint: disable=too-many-arguments,too-many-locals
    model:R3D,
    images:torch.Tensor,
    voxels:torch.Tensor,
    vae_optimizer:torch.optim.Optimizer,
    disc_optimizer:torch.optim.Optimizer,
    kl_weight=1.0,
    adv_weight=1e-2,
    pos_weight=1.0,
)->Dict[str, float]:
    """One discriminator update followed by one encoder/generator update

    Args:
        images (torch.Tensor): (batch, view, channel, height, width)
        voxels (torch.Tensor): (batch, D, H, W) occupancy in [0, 1]

    Returns:
        dict[str, float]: losses of this step
    """
    voxels = voxels.unsqueeze(1)
    recon_logits, kl_val = model(images)
    fake = torch.sigmoid(recon_logits)

    d_loss = (
        gan_loss(model.discriminator(voxels), True)
        + gan_loss(model.discriminator(fake.detach()), False)
    )
    disc_optimizer.zero_grad()
    d_loss.backward()
    disc_optimizer.step()

    recon = vae_loss(recon_logits, voxels, pos_weight)
    kl_scaled = kl_val / voxels[0].numel()
    adv = gan_loss(model.discriminator(fake), True)
    g_loss = recon + kl_weight * kl_scaled + adv_weight * adv
    vae_optimizer.zero_grad()
    g_loss.backward()
    vae_optimizer.step()

    return {
        "recon": recon.item(),
        "kl": kl_val.item(),
        "adv": adv.item(),
        "gen": g_loss.item(),
        "disc": d_loss.item(),
    }


def train_epoch( # pylint: disable=too-many-arguments
    model:R3D,
    loader:DataLoader,
    vae_optimizer:torch.optim.Optimizer,
    disc_optimizer:torch.optim.Optimizer,
    device:torch.device,
    kl_weight=1.0,
    adv_weight=1e-2,
    pos_weight=1.0,
)->Dict[str, float]:
    """
    Returns:
        dict[str, float]: losses averaged over the epoch
    """
    model.train()
    totals: Dict[str, float] = {}
    for images, voxels in loader:
        losses = train_step(
            model, images.to(device), voxels.to(device),
            vae_optimizer, disc_optimizer, kl_weight, adv_weight, pos_weight
        )
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / max(len(loader), 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model:R3D, loader:DataLoader, device:torch.device)->Dict[str, float]:
    """
    Returns:
        dict[str, float]: mean reconstruction loss and voxel IoU over the loader
    """
    model.eval()
    recon_total, iou_total, count = 0.0, 0.0, 0
    for images, voxels in loader:
        images, voxels = images.to(device), voxels.to(device)
        recon_logits, _ = model(images)
        recon_total += vae_loss(recon_logits, voxels.unsqueeze(1)).item() * len(voxels)
        iou_total += voxel_iou(torch.sigmoid(recon_logits).squeeze(1), voxels).sum().item()
        count += len(voxels)
    return {"recon": recon_total / max(count, 1), "iou": iou_total / max(count, 1)}
