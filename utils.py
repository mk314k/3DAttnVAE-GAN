"""
This module includes all the utilities code like displaying images, loading data
from shapenet core dataset.
Author:mk314k
"""
import os
from typing import Tuple
import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt


def voxshow(image):
    """
    Display 3d image
    Args:
        img (torch.Tensor|np.ndarray): 3d image to display
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([0, image.shape[0]])
    ax.set_ylim([0, image.shape[1]])
    ax.set_zlim([0, image.shape[2]])
    ax.voxels(image, edgecolor="b")
    plt.show()


def load_views(folder_path:str, num_views:int, image_size:Tuple[int, int])->torch.Tensor:
    """
    Reads views 000.png, 001.png, ... of one object as grayscale

    Args:
        folder_path (str): folder containing the rendered views
        num_views (int): number of views to read
        image_size (tuple[int, int]): (height, width) to resize each view to

    Returns:
        torch.Tensor: (view, 1, height, width) in [0, 1]
    """
    height, width = image_size
    views = []
    for view in range(num_views):
        path = os.path.join(folder_path, f"{view:03d}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # pylint: disable=no-member
        if img is None:
            raise FileNotFoundError(f"could not read view {path}")
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)  # pylint: disable=no-member
        views.append(img)
    return torch.from_numpy(np.stack(views)).float().div_(255.0).unsqueeze(1)


def downsample(mat:np.ndarray, new_size=64)->np.ndarray:
    """
    Args:
        mat (np.ndarray): cubic 3d image (N, N, N)
        new_size (int, optional): the new number of voxels cubes. Defaults to 64.

    Returns:
        np.ndarray: 3d image (new_size, new_size, new_size) after average pooling
    """
    if mat.shape[0] % new_size:
        raise ValueError(f"voxel grid of size {mat.shape[0]} is not divisible by {new_size}")
    factor = mat.shape[0] // new_size
    return mat.reshape(
        new_size, factor, new_size, factor, new_size, factor
    ).mean(axis=(1, 3, 5))


def load_voxels(mat_file_path:str, voxel_size=None)->torch.Tensor:
    """
    Args:
        mat_file_path (str): path of a model.mat file with an "input" voxel array
        voxel_size (int, optional): downsample the grid to this size

    Returns:
        torch.Tensor: occupancy grid (D, H, W) in [0, 1]
    """
    mat = np.squeeze(sio.loadmat(mat_file_path)["input"]).astype(np.float32)
    if voxel_size is not None and mat.shape[0] != voxel_size:
        mat = downsample(mat, voxel_size)
    return torch.from_numpy(np.ascontiguousarray(mat))


class ShapeNetVoxelDataset(Dataset):
    """
    Pairs of 2d views and 3d voxel grids, read lazily from disk.

    Expected layout (as downloaded from the shapenet core dataset website):
        <root>/<split>_imgs/<object>/000.png ... 011.png
        <root>/<split>_voxels/<object>/model.mat
    """
    def __init__(self, data_path:str, image_size=(256, 192), voxel_size=64, num_views=12):
        if not data_path.rstrip("/").endswith("imgs"):
            raise ValueError(f"data_path must end with 'imgs', got {data_path}")
        self.data_path = data_path.rstrip("/")
        self.voxel_path = self.data_path[:-4] + "voxels"
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.num_views = num_views
        self.objects = sorted(
            name for name in os.listdir(self.data_path)
            if not name.startswith(".") and os.path.isdir(os.path.join(self.data_path, name))
        )

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index:int)->Tuple[torch.Tensor, torch.Tensor]:
        name = self.objects[index]
        views = load_views(
            os.path.join(self.data_path, name), self.num_views, self.image_size
        )
        voxels = load_voxels(
            os.path.join(self.voxel_path, name, "model.mat"), self.voxel_size
        )
        return views, voxels
