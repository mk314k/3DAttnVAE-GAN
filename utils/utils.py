"""
This module includes all the utilities code used by other modules
"""
import os
import torch
from matplotlib import pyplot as plt
import cv2

# from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import numpy as np


# def imshow3(n):
#     plt.imshow(img2d1[n,2].permute((1,2,0)))


# def imshow(img2d, n):
#     """ """
#     plt.imshow(img2d[n, 2])


def voxshow(image):
    """_summary_

    Args:
        img (_type_): _description_
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([0, image.shape[0]])
    ax.set_ylim([0, image.shape[1]])
    ax.set_zlim([0, image.shape[2]])
    ax.voxels(image, edgecolor="b")
    plt.show()


gray_filt = np.array([0.299, 0.587, 0.114])


def preprocess(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return torch.tensor(x @ gray_filt) # pylint: disable=no-member


def downsample(mat, down_sample=64):
    """_summary_

    Args:
        mat (_type_): _description_
        downsample (int, optional): _description_. Defaults to 64.

    Returns:
        _type_: _description_
    """
    return torch.tensor( # pylint: disable=no-member
        mat.reshape((-1, down_sample, 4, down_sample, 4, down_sample, 4)).mean(
            axis=(2, 4, 6)
        )
    )

if __name__ == "__main__":
    DATA_PATH = "hundred/img"
    img2d = []
    img3d = []
    for i, folder_name in enumerate(os.listdir(DATA_PATH)):
        folder_path = os.path.join(DATA_PATH, folder_name)
        if folder_name[0] != ".":
            mat_file_path = os.path.join(f"hundred/vox/{folder_name}/model.mat")
            mat_data = sio.loadmat(mat_file_path)["input"][0]
            downsampled_data = downsample(mat_data)
            img = []
            for i in range(12):
                img.append(cv2.imread(folder_path + "/" + format(i, "03d") + ".png")) # pylint: disable=no-member
            img2d.append(preprocess(np.array(img)))
            img3d.append(downsampled_data)
