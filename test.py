"""
Smoke tests for 3DAttnVAE-GAN, run with `pytest test.py`
"""
import os
import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from models.r3d import R3D
from train import train_epoch, evaluate, voxel_iou
from utils import ShapeNetVoxelDataset, downsample

IMAGE_SIZE = (64, 48)
NUM_VIEWS = 3


def small_model() -> R3D:
    """Model sized for fast CPU tests"""
    return R3D(image_size=IMAGE_SIZE, embedding_dim=32, num_layers=1, latent_dim=64)


def test_forward_shapes():
    """Encoder, generator and discriminator produce the expected shapes"""
    model = small_model()
    images = torch.rand(2, NUM_VIEWS, 1, *IMAGE_SIZE)
    logits, kl_val = model(images)
    assert logits.shape == (2, 1, 64, 64, 64)
    assert kl_val.dim() == 0 and kl_val.item() >= 0
    assert model.discriminator(torch.sigmoid(logits)).shape == (2, 1)


def test_downsample():
    """Average pooling keeps occupancy fractions"""
    mat = np.zeros((128, 128, 128), dtype=np.float32)
    mat[:2, :2, :2] = 1
    out = downsample(mat, 64)
    assert out.shape == (64, 64, 64)
    assert out[0, 0, 0] == 1 and out.sum() == 1


def test_voxel_iou():
    """IoU is 1 for identical grids and 0 for disjoint ones"""
    grid = torch.zeros(2, 4, 4, 4)
    grid[0, 0] = 1
    grid[1, 1] = 1
    assert torch.allclose(voxel_iou(grid, grid), torch.ones(2))
    assert torch.allclose(voxel_iou(grid, grid.flip(0)), torch.zeros(2))


def make_fake_dataset(root, num_objects=4) -> str:
    """Writes a tiny dataset in the shapenet core layout and returns the imgs path"""
    img_path = os.path.join(root, "train_imgs")
    for index in range(num_objects):
        name = f"obj{index}"
        os.makedirs(os.path.join(img_path, name))
        os.makedirs(os.path.join(root, "train_voxels", name))
        for view in range(NUM_VIEWS):
            img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(img_path, name, f"{view:03d}.png"), img)  # pylint: disable=no-member
        voxels = (np.random.rand(1, 128, 128, 128) > 0.9).astype(np.uint8)
        sio.savemat(os.path.join(root, "train_voxels", name, "model.mat"), {"input": voxels})
    return img_path


def test_train_and_evaluate(tmp_path):
    """One epoch of training on files read from disk updates every sub-network"""
    torch.manual_seed(0)
    dataset = ShapeNetVoxelDataset(
        make_fake_dataset(str(tmp_path)), image_size=IMAGE_SIZE, voxel_size=64, num_views=NUM_VIEWS
    )
    views, voxels = dataset[0]
    assert views.shape == (NUM_VIEWS, 1, *IMAGE_SIZE)
    assert voxels.shape == (64, 64, 64)

    loader = DataLoader(dataset, batch_size=2)
    model = small_model()
    vae_optim = torch.optim.AdamW(model.vae_parameters(), lr=1e-3)
    disc_optim = torch.optim.AdamW(model.disc_parameters(), lr=1e-3)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    losses = train_epoch(model, loader, vae_optim, disc_optim, torch.device("cpu"))
    assert all(np.isfinite(value) for value in losses.values())
    for part in ("encoder", "generator", "discriminator"):
        assert any(
            not torch.equal(p, before[name])
            for name, p in model.named_parameters() if name.startswith(part)
        ), f"{part} was not updated"

    metrics = evaluate(model, loader, torch.device("cpu"))
    assert 0.0 <= metrics["iou"] <= 1.0
