"""
This module includes all the utilities code like displaying images, loading data
from shapenet core dataset.
Author:mk314k
"""
import os
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import cv2
# from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import numpy as np



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



def preprocess(images, sizex=None, sizey=None):
    """
    Preprocesses the input images by converting them to grayscale and resizing 
    them if sizex and sizey are provided.

    Args:
        images (np.ndarray): Input images (batch, channels, height, width).
        sizex (int, optional): New width for resizing (default: None).
        sizey (int, optional): New height for resizing (default: None).

    Returns:
        torch.Tensor: Preprocessed images as a tensor.
    """

    images_tensor = torch.from_numpy(images) # pylint: disable=no-member
    transform_list = [transforms.Grayscale()]
    if sizex and sizey:
        transform_list.append(transforms.Resize((sizey, sizex)))
    composed_transforms = transforms.Compose(transform_list)
    transformed_images = composed_transforms(images_tensor)

    return transformed_images



def downsample(mat:np.ndarray, new_size=64)->np.ndarray:
    """
    Args:
        mat (np.ndarray): 3d image 
        new_size (int, optional): the new number of voxels cubes. Defaults to 64.

    Returns:
        np.ndarray: 3d images after downsampling
    """
    lost_dim = int(mat.shape[2]/new_size)
    return mat.reshape((-1, new_size, lost_dim, new_size, lost_dim, new_size, lost_dim)).mean(
            axis=(2, 4, 6)
        )


def load_data(data_path:str, down_sample=None):
    """
    Load 2d and 3d images from given path
    Args:
        data_path (string): path of image files, it must ends with imgs
                            as downloded from shapenet core dataset website
        down_sample (int, optional): Downsamples 3d images if set to an integer.
                                     Defaults to None.

    Returns:
        tuple[List[torch.Tesnor]]: 2d and their corresponding 3d images in tensor form
    """
    img2d = []
    img3d = []
    for i, folder_name in enumerate(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder_name)
        if folder_name[0] != ".":
            mat_file_path = os.path.join(f"{data_path[:-4]}voxels/{folder_name}/model.mat")
            mat_data = sio.loadmat(mat_file_path)["input"][0]
            if down_sample is not None:
                mat_data = downsample(mat_data, down_sample)
            mat_data = torch.tensor(mat_data) # pylint: disable=no-member
            img = []
            for i in range(12): #shapenet core has 12 views of 2d images
                img.append(cv2.imread(folder_path + "/" + format(i, "03d") + ".png")) # pylint: disable=no-member
            img2d.append(preprocess(np.array(img)))
            img3d.append(mat_data)
    return img2d, img3d
