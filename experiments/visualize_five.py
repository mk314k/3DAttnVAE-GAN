"""
Renders reconstructions of the overfit_five checkpoint on seen and unseen views,
without any training.

Run from the repo root after experiments.overfit_five:
    python -m experiments.visualize_five
"""
import argparse
import os
import cv2
import numpy as np
from scipy import ndimage
import torch
from PIL import Image
from models.r3d import R3D
from train import voxel_iou
from utils import ShapeNetVoxelDataset
from experiments.overfit_five import (
    GRID, HELD_OUT_VIEWS, IMAGE_SIZE, NUM_VIEWS, render, write_dataset
)

SEEN_VIEWS = (0, 4, 8)
TILE = 128
DISPLAY_AZIMUTH = 30


def label(text: str, width: int) -> np.ndarray:
    """White strip with centred text"""
    strip = np.full((24, width), 255, dtype=np.uint8)
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]  # pylint: disable=no-member
    cv2.putText(  # pylint: disable=no-member
        strip, text, ((width - size[0]) // 2, 17),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1, cv2.LINE_AA  # pylint: disable=no-member
    )
    return strip


def voxel_tile(voxels: np.ndarray, azimuth=DISPLAY_AZIMUTH) -> np.ndarray:
    """Renders an (x, y, z) occupancy grid from the given azimuth"""
    return cv2.resize(  # pylint: disable=no-member
        render(voxels.transpose(2, 1, 0) >= 0.5, azimuth), (TILE, TILE)
    )


@torch.no_grad()
def build_turntable(model, dataset, view: int, path: str, num_frames=36):
    """
    Saves a GIF with one row per shape: the single input view, then the ground truth
    and the reconstruction rotating side by side
    """
    rows = []
    for images, voxels in dataset:
        pred = model.reconstruct(images[view:view + 1][None])[0].numpy()
        image = cv2.resize((images[view, 0].numpy() * 255).astype(np.uint8), (TILE, TILE))  # pylint: disable=no-member
        rows.append((image, voxels.numpy(), pred))

    header = np.hstack([
        label("unseen input", TILE), label("ground truth", TILE),
        label("reconstruction", TILE),
    ])
    frames = []
    for step in range(num_frames):
        azimuth = step * 360 / num_frames
        body = [
            np.hstack([image, voxel_tile(truth, azimuth), voxel_tile(pred, azimuth)])
            for image, truth, pred in rows
        ]
        frames.append(Image.fromarray(np.vstack([header] + body)))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=80, loop=0)


@torch.no_grad()
def build_grid(model, dataset, views, title: str):  # pylint: disable=too-many-locals
    """
    One row per shape: ground truth, then (input view, prediction) for every view

    Returns:
        tuple[np.ndarray, dict[str, list[float]]]: the image and IoU per view for every shape
    """
    header = [label("ground truth", TILE)]
    for view in views:
        header += [label(f"input view {view}", TILE), label("prediction", TILE)]
    rows, ious = [np.hstack(header)], {}
    for index, name in enumerate(dataset.objects):
        images, voxels = dataset[index]
        row, ious[name] = [voxel_tile(voxels.numpy())], []
        for view in views:
            pred = model.reconstruct(images[view:view + 1][None])
            ious[name].append(voxel_iou(pred, voxels[None]).item())
            image = (images[view, 0].numpy() * 255).astype(np.uint8)
            row += [cv2.resize(image, (TILE, TILE)), voxel_tile(pred[0].numpy())]  # pylint: disable=no-member
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    return np.vstack([label(title, grid.shape[1]), grid]), ious


@torch.no_grad()
def build_display(  # pylint: disable=too-many-arguments,too-many-locals
    model, dataset, view: int, path: str, size=320, num_frames=36
):
    """
    Saves a GIF that shows each input image, then its reconstruction rotating with the
    input as an inset, one shape after another
    """
    frames, durations = [], []
    start = view * 360 / NUM_VIEWS
    for images, _ in dataset:
        image = (images[view, 0].numpy() * 255).astype(np.uint8)
        pred = model.reconstruct(images[view:view + 1][None])[0].numpy()
        smooth = ndimage.zoom(pred, GRID / pred.shape[0], order=1) >= 0.5
        frames.append(np.vstack([
            label("input: a single 2D image", size),
            cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA),  # pylint: disable=no-member
        ]))
        durations.append(1500)
        inset = cv2.resize(image, (size // 4, size // 4), interpolation=cv2.INTER_AREA)  # pylint: disable=no-member
        for step in range(num_frames):
            panel = cv2.resize(  # pylint: disable=no-member
                render(smooth.transpose(2, 1, 0), start + step * 360 / num_frames), (size, size),
                interpolation=cv2.INTER_AREA,  # pylint: disable=no-member
            )
            panel[4:4 + inset.shape[0], 4:4 + inset.shape[1]] = inset
            frames.append(np.vstack([label("3D reconstruction", size), panel]))
            durations.append(70)
    images_out = [Image.fromarray(frame) for frame in frames]
    images_out[0].save(
        path, save_all=True, append_images=images_out[1:], duration=durations, loop=0
    )


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="experiments/out/model.pt")
    parser.add_argument("--data", default="experiments/out/data")
    parser.add_argument("--out", default="assets")
    args = parser.parse_args()

    img_root = os.path.join(args.data, "train_imgs")
    if not os.path.isdir(img_root):
        write_dataset(args.data)
    dataset = ShapeNetVoxelDataset(
        img_root, image_size=IMAGE_SIZE, voxel_size=64, num_views=NUM_VIEWS
    )
    model = R3D(image_size=IMAGE_SIZE, latent_dim=256)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    os.makedirs(args.out, exist_ok=True)
    for views, kind in ((SEEN_VIEWS, "seen"), (HELD_OUT_VIEWS, "unseen")):
        title = f"{kind} views ({'used' if kind == 'seen' else 'never used'} in training)"
        grid, ious = build_grid(model, dataset, views, title)
        path = os.path.join(args.out, f"overfit_{kind}.png")
        cv2.imwrite(path, grid)  # pylint: disable=no-member
        print(f"{path}")
        for name, values in ious.items():
            per_view = "  ".join(f"view {v}: {iou:.3f}" for v, iou in zip(views, values))
            print(f"  {name:7s} {per_view}")
        print(f"  mean IoU {np.mean([v for vals in ious.values() for v in vals]):.3f}")

    path = os.path.join(args.out, "turntable.gif")
    build_turntable(model, dataset, HELD_OUT_VIEWS[1], path)
    print(path)
    build_display(model, dataset, HELD_OUT_VIEWS[1], "display.gif")
    print("display.gif")


if __name__ == "__main__":
    main()
