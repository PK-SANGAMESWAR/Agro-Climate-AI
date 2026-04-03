import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Config
IMAGE_SIZE = 64
BATCH_SIZE = 32
LATENT_DIM_GAN = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader():
    dataset_path = "data/nonsegmentedv2"
    if not os.path.exists(dataset_path):
        dataset_path = "data" 

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader, dataset.classes

# --- Model Definitions ---

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh()
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM_GAN, 512, 4, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )
    def forward(self, x): return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512*4*4, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.model(x)

def visualize_latent_space(ae, dataloader, device):
    ae.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            _, z = ae(imgs.to(device))
            latents.append(z.view(imgs.size(0), -1).cpu().numpy())
            labels.append(lbls.numpy())
    
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # PCA
    pca = PCA(n_components=2)
    lat_pca = pca.fit_transform(latents)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30)
    lat_tsne = tsne.fit_transform(latents[:1000]) # subset for speed
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(lat_pca[:, 0], lat_pca[:, 1], c=labels, cmap='tab10', alpha=0.5)
    plt.title("PCA of Latent Space")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.scatter(lat_tsne[:, 0], lat_tsne[:, 1], c=labels[:1000], cmap='tab10', alpha=0.5)
    plt.title("t-SNE of Latent Space")
    plt.colorbar()
    plt.savefig("models/latent_space.png")
    plt.show()

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    dataloader, classes = get_dataloader()
    
    ae = Autoencoder().to(DEVICE)
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    
    # --- Quick Training Logic or Load Pre-trained weights ---
    print("Training Autoencoder (short run)...")
    opt_ae = optim.Adam(ae.parameters(), lr=0.001)
    crit_ae = nn.MSELoss()
    for ep in range(2):
        for imgs, _ in dataloader:
            imgs = imgs.to(DEVICE)
            out, _ = ae(imgs)
            loss = crit_ae(out, imgs)
            opt_ae.zero_grad()
            loss.backward()
            opt_ae.step()
        print(f"AE Epoch {ep+1} complete")
    
    torch.save(ae.state_dict(), "models/ae_model.pth")
    visualize_latent_space(ae, dataloader, DEVICE)
    
    print("Training GAN (short run)...")
    opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    crit_gan = nn.BCELoss()
    for ep in range(2):
        for imgs, _ in dataloader:
            bs = imgs.size(0)
            real_lbl = torch.ones(bs, 1).to(DEVICE)
            fake_lbl = torch.zeros(bs, 1).to(DEVICE)
            
            # D
            z = torch.randn(bs, LATENT_DIM_GAN, 1, 1).to(DEVICE)
            fake = G(z)
            l_real = crit_gan(D(imgs.to(DEVICE)), real_lbl)
            l_fake = crit_gan(D(fake.detach()), fake_lbl)
            (l_real + l_fake).backward()
            opt_D.step()
            opt_D.zero_grad()
            
            # G
            l_G = crit_gan(D(fake), real_lbl)
            l_G.backward()
            opt_G.step()
            opt_G.zero_grad()
        print(f"GAN Epoch {ep+1} complete")
        
    torch.save(G.state_dict(), "models/gan_generator.pth")
    print("Models saved in 'models/'")
