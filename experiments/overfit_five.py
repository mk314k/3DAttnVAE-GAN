"""
Overfitting experiment: can the model reconstruct 5 shapes from a single view?

Generates 5 procedural shapes in the shapenet core layout, trains single-view
(one image in, one voxel grid out) on 9 of the 12 views of each shape and
evaluates on the training views and on the 3 held-out views.

Run from the repo root: python -m experiments.overfit_five --epochs 150
"""
import argparse
import os
import time
import cv2
import numpy as np
import scipy.io as sio
from scipy import ndimage
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.r3d import R3D
from train import train_epoch, evaluate, voxel_iou
from utils import ShapeNetVoxelDataset

GRID = 128
NUM_VIEWS = 12
IMAGE_SIZE = (128, 128)
HELD_OUT_VIEWS = (2, 6, 10)


def _box(x, y, z, lo, hi):
    return (
        (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1]) & (z >= lo[2]) & (z <= hi[2])
    )


def make_shapes() -> dict:
    """
    Returns:
        dict[str, np.ndarray]: boolean occupancy grids (z, y, x) with z pointing up
    """
    axis = np.linspace(-1, 1, GRID)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    radius = np.sqrt(x ** 2 + y ** 2)
    shapes = {
        "sphere": x ** 2 + y ** 2 + z ** 2 < 0.6 ** 2,
        "torus": (radius - 0.5) ** 2 + z ** 2 < 0.18 ** 2,
    }

    chair = _box(x, y, z, (-0.45, -0.45, -0.05), (0.45, 0.45, 0.05))
    chair |= _box(x, y, z, (-0.45, 0.35, 0.05), (0.45, 0.45, 0.75))
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx * 0.39, sy * 0.39
            chair |= _box(x, y, z, (cx - 0.05, cy - 0.05, -0.7), (cx + 0.05, cy + 0.05, -0.05))
    shapes["chair"] = chair

    table = _box(x, y, z, (-0.75, -0.45, 0.3), (0.75, 0.45, 0.4))
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx * 0.66, sy * 0.37
            table |= _box(x, y, z, (cx - 0.05, cy - 0.05, -0.7), (cx + 0.05, cy + 0.05, 0.3))
    shapes["table"] = table

    mug = (radius >= 0.3) & (radius <= 0.4) & (np.abs(z) <= 0.5)
    mug |= (radius <= 0.4) & (z >= -0.5) & (z <= -0.42)
    handle = (np.sqrt((x - 0.42) ** 2 + z ** 2) - 0.22) ** 2 + y ** 2 < 0.06 ** 2
    shapes["mug"] = mug | (handle & (x > 0.38))
    return shapes


def render(grid: np.ndarray, azimuth: float, elevation=25.0) -> np.ndarray:
    """
    Orthographic depth-shaded render on a white background

    Args:
        grid (np.ndarray): boolean occupancy (z, y, x)
        azimuth (float): rotation around the vertical axis in degrees
        elevation (float): camera elevation in degrees

    Returns:
        np.ndarray: uint8 grayscale image (GRID, GRID)
    """
    rotated = ndimage.rotate(grid.astype(np.uint8), azimuth, axes=(1, 2), reshape=False, order=0)
    rotated = ndimage.rotate(rotated, elevation, axes=(0, 1), reshape=False, order=0) > 0
    hit = rotated.any(axis=1)
    depth = np.where(hit, rotated.argmax(axis=1), GRID).astype(np.float32)
    grad_z, grad_x = np.gradient(np.where(hit, depth, depth[hit].max() if hit.any() else 0))
    normal = np.stack([-grad_x, -grad_z, np.ones_like(depth)])
    normal /= np.linalg.norm(normal, axis=0, keepdims=True)
    light = np.array([0.4, 0.5, 0.75])
    lambert = np.clip(np.tensordot(light / np.linalg.norm(light), normal, axes=1), 0, 1)
    shade = 0.15 + 0.55 * lambert + 0.25 * (1 - depth / GRID)
    image = np.where(hit, shade, 1.0)[::-1]
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def write_dataset(root: str) -> str:
    """Writes renders and voxels in the shapenet core layout, returns the imgs path"""
    img_root = os.path.join(root, "train_imgs")
    for name, grid in make_shapes().items():
        os.makedirs(os.path.join(img_root, name), exist_ok=True)
        os.makedirs(os.path.join(root, "train_voxels", name), exist_ok=True)
        for view in range(NUM_VIEWS):
            image = render(grid, azimuth=view * 360 / NUM_VIEWS)
            cv2.imwrite(os.path.join(img_root, name, f"{view:03d}.png"), image)  # pylint: disable=no-member
        voxels = grid.transpose(2, 1, 0)[None].astype(np.uint8)
        sio.savemat(os.path.join(root, "train_voxels", name, "model.mat"), {"input": voxels})
    return img_root


def single_view_split(dataset: ShapeNetVoxelDataset):
    """
    Turns each (12 views, voxels) object into 12 single-view samples

    Returns:
        tuple[TensorDataset, TensorDataset, list[str]]: train and held-out samples,
        plus the shape name for every held-out sample
    """
    train_x, train_y, test_x, test_y, test_names = [], [], [], [], []
    for index, name in enumerate(dataset.objects):
        views, voxels = dataset[index]
        for view in range(NUM_VIEWS):
            sample = views[view:view + 1]
            if view in HELD_OUT_VIEWS:
                test_x.append(sample)
                test_y.append(voxels)
                test_names.append(name)
            else:
                train_x.append(sample)
                train_y.append(voxels)
    return (
        TensorDataset(torch.stack(train_x), torch.stack(train_y)),
        TensorDataset(torch.stack(test_x), torch.stack(test_y)),
        test_names,
    )


@torch.no_grad()
def shape_recognition(model, dataset, names, shapes: dict) -> float:
    """
    Fraction of held-out views whose prediction overlaps its own shape more than any other
    """
    model.eval()
    names_sorted = sorted(shapes)
    targets = torch.stack([shapes[name] for name in names_sorted])
    correct = 0
    for (image, _), name in zip(dataset, names):
        pred = model.reconstruct(image[None])
        ious = voxel_iou(pred.expand(len(targets), -1, -1, -1), targets)
        correct += names_sorted[int(ious.argmax())] == name
    return correct / len(names)


def save_comparison(model, dataset, names, path: str):
    """Saves input view | ground truth render | prediction render for one view per shape"""
    model.eval()
    rows, seen = [], set()
    for (image, voxels), name in zip(dataset, names):
        if name in seen:
            continue
        seen.add(name)
        pred = model.reconstruct(image[None])[0].numpy() >= 0.5
        size = (GRID, GRID)
        tiles = [
            cv2.resize((image[0, 0].numpy() * 255).astype(np.uint8), size),  # pylint: disable=no-member
            cv2.resize(render(voxels.numpy().transpose(2, 1, 0) >= 0.5, 30), size),  # pylint: disable=no-member
            cv2.resize(render(pred.transpose(2, 1, 0), 30), size),  # pylint: disable=no-member
        ]
        rows.append(np.hstack(tiles))
    cv2.imwrite(path, np.vstack(rows))  # pylint: disable=no-member


def main():  # pylint: disable=too-many-locals
    """Experiment entry point"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--out", default="experiments/out")
    args = parser.parse_args()
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    dataset = ShapeNetVoxelDataset(
        write_dataset(os.path.join(args.out, "data")),
        image_size=IMAGE_SIZE, voxel_size=64, num_views=NUM_VIEWS,
    )
    train_set, test_set, test_names = single_view_split(dataset)
    shapes = {name: dataset[i][1] for i, name in enumerate(dataset.objects)}
    print(f"{len(train_set)} training views, {len(test_set)} held-out views, {len(shapes)} shapes")

    targets = torch.stack(list(shapes.values()))
    baseline = voxel_iou(targets.mean(0).expand(len(shapes), -1, -1, -1), targets)
    print(f"baseline IoU of always predicting the average shape: {baseline.mean():.3f}")

    device = torch.device("cpu")
    model = R3D(image_size=IMAGE_SIZE, latent_dim=256)
    vae_optim = torch.optim.AdamW(model.vae_parameters(), lr=1e-3, weight_decay=0.01)
    disc_optim = torch.optim.AdamW(model.disc_parameters(), lr=1e-4, weight_decay=0.01)
    train_loader = DataLoader(train_set, batch_size=9, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=15)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(
            model, train_loader, vae_optim, disc_optim, device,
            kl_weight=0.1, adv_weight=1e-3, pos_weight=20.0
        )
        if epoch % 10 == 0 or epoch == 1:
            train_iou = evaluate(model, train_loader, device)["iou"]
            test_iou = evaluate(model, test_loader, device)["iou"]
            print(
                f"epoch {epoch:4d}  recon {losses['recon']:.4f}  kl {losses['kl']:7.1f}  "
                f"train IoU {train_iou:.3f}  held-out-view IoU {test_iou:.3f}  "
                f"({time.time() - start:.0f}s)", flush=True
            )

    accuracy = shape_recognition(model, test_set, test_names, shapes)
    print(f"held-out views reconstructed as the correct shape: {accuracy:.0%}")
    save_comparison(model, test_set, test_names, os.path.join(args.out, "comparison.png"))
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()
