"""
Trains 3DAttnVAE-GAN on the shapenet core dataset
"""
import argparse
import os
from dataclasses import asdict
import torch
from torch.utils.data import DataLoader
import tqdm.auto as tqdm
from matplotlib import pyplot as plt
from config import Config
from models.r3d import R3D
from train import train_epoch, evaluate
from utils import ShapeNetVoxelDataset


def parse_args() -> Config:
    """
    Returns:
        Config: defaults from config.py overridden by command line arguments
    """
    cfg = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default=cfg.train_data_path)
    parser.add_argument("--test-data", default=cfg.test_data_path)
    parser.add_argument("--epochs", type=int, default=cfg.num_epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--checkpoint-dir", default=cfg.checkpoint_dir)
    args = parser.parse_args()
    cfg.train_data_path = args.train_data
    cfg.test_data_path = args.test_data
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.checkpoint_dir = args.checkpoint_dir
    return cfg


def make_loader(cfg: Config, data_path: str, shuffle: bool) -> DataLoader:
    """
    Returns:
        DataLoader: batches of (views, voxels)
    """
    dataset = ShapeNetVoxelDataset(
        data_path, image_size=cfg.image_size, voxel_size=cfg.voxel_size, num_views=cfg.num_views
    )
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=2, drop_last=shuffle, pin_memory=torch.cuda.is_available()
    )


def main():
    """Training entry point"""
    cfg = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = make_loader(cfg, cfg.train_data_path, shuffle=True)
    test_loader = make_loader(cfg, cfg.test_data_path, shuffle=False)

    model = R3D(
        in_channel=cfg.in_channel,
        image_size=cfg.image_size,
        embedding_dim=cfg.embedding_dim,
        embedding_kernel=cfg.embedding_kernel,
        attention_head=cfg.attention_head,
        num_layers=cfg.num_layers,
        latent_dim=cfg.latent_dim,
    ).to(device)

    vae_optim = torch.optim.AdamW(
        model.vae_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas
    )
    disc_optim = torch.optim.AdamW(
        model.disc_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas
    )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    history = []
    best_iou = -1.0
    for epoch in tqdm.trange(cfg.num_epochs):
        train_metrics = train_epoch(
            model, train_loader, vae_optim, disc_optim, device,
            cfg.kl_weight, cfg.adv_weight, cfg.pos_weight
        )
        test_metrics = evaluate(model, test_loader, device)
        history.append((train_metrics, test_metrics))
        tqdm.tqdm.write(
            f"epoch {epoch + 1}: train {train_metrics} | "
            f"test recon {test_metrics['recon']:.4f} iou {test_metrics['iou']:.4f}"
        )
        state = {"epoch": epoch + 1, "model": model.state_dict(), "config": asdict(cfg)}
        torch.save(state, os.path.join(cfg.checkpoint_dir, "last.pt"))
        if test_metrics["iou"] > best_iou:
            best_iou = test_metrics["iou"]
            torch.save(state, os.path.join(cfg.checkpoint_dir, "best.pt"))

    plt.figure(figsize=(16, 6))
    epochs = range(1, len(history) + 1)
    plt.plot(epochs, [train["gen"] for train, _ in history], label='Training VAE loss')
    plt.plot(epochs, [train["disc"] for train, _ in history], label='Training GAN loss')
    plt.plot(epochs, [test["iou"] for _, test in history], label='Test IoU')
    plt.xlabel('Epoch')
    plt.title('Model Performance')
    plt.legend()
    plt.savefig(os.path.join(cfg.checkpoint_dir, "training.png"))


if __name__ == "__main__":
    main()
