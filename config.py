"""
Default hyperparameters for 3DAttnVAE-GAN
Author:mk314k
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes
    """Model, data and optimisation settings"""
    train_data_path: str = "data/shapenetcore/train_imgs"
    test_data_path: str = "data/shapenetcore/test_imgs"
    image_size: Tuple[int, int] = (256, 192)  # (height, width)
    num_views: int = 12
    voxel_size: int = 64

    in_channel: int = 1
    embedding_dim: int = 64
    embedding_kernel: int = 3
    attention_head: int = 4
    num_layers: int = 2
    latent_dim: int = 1024

    batch_size: int = 8
    num_epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 0.2
    betas: Tuple[float, float] = (0.9, 0.98)
    kl_weight: float = 1.0
    adv_weight: float = 1e-2
    pos_weight: float = 20.0
    checkpoint_dir: str = "checkpoints"
