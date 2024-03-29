"""
This module includes all the loss and other functions necessary for training the model
Author:mk314k
"""
# from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from models.r3d import R3D


def gan_loss(discriminator_output:torch.Tensor, is_real:bool)->torch.Tensor:
    """Measure to classify real 3d image with generated 3d image

    Args:
        discriminator_output (torch.Tensor): _description_
        is_real (bool): _description_

    Returns:
        torch.Tensor: _description_
    """
    if is_real:
        target = torch.ones_like(discriminator_output) # pylint: disable=no-member
    else:
        target = torch.zeros_like(discriminator_output) # pylint: disable=no-member
    loss = F.binary_cross_entropy(discriminator_output, target, reduction='mean')
    return loss


def vae_loss(recon_x, label):
    """_summary_

    Args:
        recon_x (_type_): _description_
        label (_type_): _description_

    Returns:
        _type_: _description_
    """
    return F.binary_cross_entropy(recon_x[0], label, reduction='sum')


def train_epoch( # pylint: disable=too-many-locals
    train_x:torch.Tensor,
    train_y:torch.Tensor,
    model:R3D,
    vae_optimizer,
    gan_optimizer
):
    """_summary_

    Args:
        train_x (torch.Tensor): _description_
        train_y (torch.Tensor): _description_
        model_e (nn.Module): _description_
        model_d (nn.Module): _description_
        model_g (nn.Module): _description_
        optims (_type_): _description_
        num_epochs (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    gan_total_loss = 0
    disc_total_loss = 0
    for batch_data, batch_label in zip(train_x, train_y):
        pred = model(batch_data, batch_label)

        g_loss = model.kl_val + vae_loss(pred, batch_label)
        vae_optimizer.zero_grad()
        g_loss.backward()
        vae_optimizer.step()
        gan_total_loss += g_loss.item()

        d_loss = gan_loss(model.disc_true, True) + gan_loss(model.disc_false, False)
        gan_optimizer.zero_grad()
        d_loss.backward()
        gan_optimizer.step()
        disc_total_loss += d_loss.item()

    return gan_total_loss, disc_total_loss

    #     g_logit = model_g(e_logit)
    #     g_loss = model_e.kl + vae_loss(g_logit, batch_label)
    #     vae_optimizer.zero_grad()
    #     g_loss.backward(retain_graph=True)
    #     vae_optimizer.step()
    #     g_logit = model_g(e_logit.detach())
    #     d_true = model_d(batch_label.reshape((1, *batch_label.shape)))
    #     d_false = model_d(g_logit)
    #     d_loss = gan_loss(d_false, False) + gan_loss(d_true, True)
    #     gan_optimizer.zero_grad()
    #     d_loss.backward()
    #     gan_optimizer.step()
    # train_losses.append((g_loss.item(), d_loss.item()))
    # return train_losses

# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     img2d = torch.zeros(10,10)
#     img3d = torch.zeros(10,10,10)
#     tt_split = train_test_split(img2d, img3d, test_size=0.3, random_state=500)
#     train_data, test_data, train_label, test_label = tt_split
#     # Intializing models
#     MODEL_E = R3DEncoder().to(device)
#     MODEL_G = R3DGenerator(1024).to(device)
#     MODEL_D = R3Discriminator().to(device)
#     # Setting hyperparameters for your optimizers
#     LR = 1e-3
#     WD = 0.2
#     betas=(0.9, 0.98)
#     # Initializing optimizers
#     vae_optim = torch.optim.AdamW(
#         list(MODEL_E.parameters())+list(MODEL_G.parameters()),
#         lr=LR,
#         weight_decay=WD,
#         betas=betas
#     )
#     gan_optim = torch.optim.AdamW(
#         list(MODEL_D.parameters())+list(MODEL_G.parameters()),
#         lr=LR,
#         weight_decay=WD,
#         betas=betas
#     )
#     train_loss = train(train_data, train_label, (MODEL_E, MODEL_G,
# MODEL_D), (vae_optim, gan_optim))
#     # Plotting training losses
#     plt.figure(figsize=(16, 6))
#     plt.plot(range(len(train_loss)), [tloss[0] for tloss in train_loss],
#  label='Training VAE loss')
#     plt.plot(range(len(train_loss)), [tloss[1] for tloss in train_loss],
#  label='Training GAN loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Model Performance')
#     plt.legend()
#     plt.show()
