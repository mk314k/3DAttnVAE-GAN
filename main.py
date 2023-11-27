"""
_summary_
"""
import torch
from matplotlib import pyplot as plt
import tqdm.auto as tqdm
from utils import load_data
from train import train_epoch
from models.r3d import R3D


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    TRAIN_DATA_PATH = "data/shapenetcore/train_imgs"
    TEST_DATA_PATH = "data/shapenetcore/test_imgs"
    VOXEL_SIZE = 64
    pixel_shape = (192,256)

    train_2d, train_3d = load_data(
        TRAIN_DATA_PATH,
        voxel_size = VOXEL_SIZE,
        pixel_shape = pixel_shape,
        device = device
    )
    test_2d, test_3d = load_data(
        TEST_DATA_PATH,
        voxel_size = VOXEL_SIZE,
        pixel_shape = pixel_shape,
        device = device
    )


    model = R3D(

    ).to(device)

    # Setting hyperparameters for optimizers
    LR = 1e-3
    WD = 0.2
    betas=(0.9, 0.98)
    # Initializing optimizers
    vae_optim = torch.optim.AdamW(
        model.vae_parameters(),
        lr=LR,
        weight_decay=WD,
        betas=betas
    )
    gan_optim = torch.optim.AdamW(
        model.gan_parameters(),
        lr=LR,
        weight_decay=WD,
        betas=betas
    )
    NUM_EPOCHS = 25
    train_loss = []
    for _ in tqdm.tqdm(NUM_EPOCHS):
        train_loss.append(train_epoch(
            train_2d, train_3d, model, vae_optim, gan_optim
        ))
    # Plotting training losses
    plt.figure(figsize=(16, 6))
    plt.plot(range(len(train_loss)), [tloss[0] for tloss in train_loss], label='Training VAE loss')
    plt.plot(range(len(train_loss)), [tloss[1] for tloss in train_loss], label='Training GAN loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Performance')
    plt.legend()
    plt.show()
