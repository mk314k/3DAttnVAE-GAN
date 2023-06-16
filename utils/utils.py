import torch
from matplotlib import pyplot as plt
import os
import cv2
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import numpy as np


def imshow3(n):
    plt.imshow(img2d1[n,2].permute((1,2,0)))
def imshow(n):
    plt.imshow(img2d[n,2])

def voxshow(img):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, img.shape[0]])
    ax.set_ylim([0, img.shape[1]])
    ax.set_zlim([0, img.shape[2]])
    ax.voxels(img, edgecolor='b')
    plt.show()

gray_filt = np.array([0.299, 0.587, 0.114])
def preprocess(x):
  return torch.tensor(x@gray_filt)
def downsample(mat, downsample_size = 64):
  return torch.tensor(mat.reshape((-1, 64, 4, 64, 4, 64, 4)).mean(axis=(2,4,6)))

data_path = 'hundred/img' 
img2d = []
img3d = []
for i, folder_name in enumerate(os.listdir(data_path)):
    folder_path = os.path.join(data_path, folder_name)
    try:
        if (folder_name[0]!='.'):
            mat_file_path = os.path.join(f'hundred/vox/{folder_name}/model.mat')
            mat_data = sio.loadmat(mat_file_path)['input'][0]
            downsampled_data = downsample(mat_data)
            img =[]
            for i in range(12):
              img.append(cv2.imread(folder_path+'/'+ format(i,'03d')+'.png'))
            img2d.append(preprocess(np.array(img)))
            img3d.append(downsampled_data)
    except:
        print(folder_path)