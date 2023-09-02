import tqdm.auto as tqdm
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F  # Add this import for torch functions
import matplotlib.pyplot as plt

# Define your GAN loss function
def gan_loss(discriminator_output, is_real):
    if is_real:
        target = torch.ones_like(discriminator_output)
    else:
        target = torch.zeros_like(discriminator_output)
    loss = F.binary_cross_entropy(discriminator_output, target, reduction='mean')
    return loss

# Define your VAE loss function
def vae_loss(recon_x, label):
    return F.binary_cross_entropy(recon_x[0], label, reduction='sum')

# Define your training function
def train():
    train_losses = []
    for epoch in tqdm.tqdm(range(num_epochs)):
        for i in range(70):
            batch_label = train_labels[i].to(torch.float).to(device)
            for j in [2]:
                batch_data = train_data[i,j].to(torch.float).reshape(1,1,192,256).to(device)
                e_logit = model_e(batch_data)
                g_logit = model_g(e_logit)
                g_loss = model_e.kl + vae_loss(g_logit, batch_label)
                vae_optim.zero_grad()
                g_loss.backward(retain_graph=True)
                vae_optim.step()
                g_logit = model_g(e_logit.detach())
                d_true = model_d(batch_label.reshape((1, *batch_label.shape)))
                d_false = model_d(g_logit)
                d_loss = gan_loss(d_false, False) + gan_loss(d_true, True)
                gan_optim.zero_grad()
                d_loss.backward()
                gan_optim.step()

        train_losses.append((g_loss.item(), d_loss.item()))
    return train_losses

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split your data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(img2d, img3d, test_size=0.3, random_state=500)
    
    # Define and initialize your models
    model_e = R3DEncoder().cuda()
    model_g = R3DGenerator(1024).cuda()
    model_d = R3Discriminator().cuda()

    # Set hyperparameters for your optimizers
    lr=1e-3
    wd=0.2
    betas=(0.9, 0.98)

    # Initialize optimizers
    vae_optim = torch.optim.AdamW(list(model_e.parameters())+list(model_g.parameters()), lr=lr, weight_decay=wd, betas=betas)
    gan_optim = torch.optim.AdamW(list(model_d.parameters())+list(model_g.parameters()), lr=lr, weight_decay=wd, betas=betas)
    
    # Plot training losses
    plt.figure(figsize=(16, 6))
    plt.plot(range(len(train_losses)), [tloss[0] for tloss in train_losses], label='Training VAE loss')
    plt.plot(range(len(train_losses)), [tloss[1] for tloss in train_losses], label='Training GAN loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Performance')
    plt.legend()
    plt.show()
