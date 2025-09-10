#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Autoencoders Comparison: Fully Connected Autoencoder vs Variational Autoencoder

This script implements and compares two types of autoencoders:
1. Fully Connected Autoencoder for MNIST with Gaussian noise
2. Convolutional Variational Autoencoder (VAE) for face image generation

Author: VShulgin
Date: 2022-08-20
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F
from typing import Tuple, List, Optional
import zipfile
import os

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AddGaussianNoise(object):
    """Add Gaussian noise to tensor for denoising autoencoder"""
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

def prepare_mnist_data(batch_size: int = 100, noise_std: float = 0.1) -> DataLoader:
    """
    Prepare MNIST dataset with Gaussian noise for denoising autoencoder.
    
    Args:
        batch_size: Batch size for data loader
        noise_std: Standard deviation of Gaussian noise
        
    Returns:
        DataLoader for MNIST dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(std=noise_std)
    ])
    
    dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class FullyConnectedAutoencoder(nn.Module):
    """Fully Connected Autoencoder for MNIST dataset"""
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 3):
        """
        Initialize the autoencoder architecture.
        
        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent space
        """
        super(FullyConnectedAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 100),
            nn.ReLU(),
            nn.Linear(100, 300),
            nn.ReLU(),
            nn.Linear(300, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder"""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def train_autoencoder(model: nn.Module, 
                     data_loader: DataLoader, 
                     epochs: int = 10, 
                     lr: float = 0.001) -> List[float]:
    """
    Train an autoencoder model.
    
    Args:
        model: Autoencoder model to train
        data_loader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        List of training losses per epoch
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    print("Training Autoencoder...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, _) in enumerate(data_loader):
            # Flatten images
            img = data.view(data.size(0), -1).to(device)
            
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1:2d}/{epochs}] | Loss: {avg_loss:.6f}')
    
    return losses

def visualize_reconstructions(model: nn.Module, 
                            data_loader: DataLoader, 
                            num_samples: int = 10) -> None:
    """
    Visualize original and reconstructed images.
    
    Args:
        model: Trained autoencoder model
        data_loader: DataLoader for test data
        num_samples: Number of samples to visualize
    """
    model.eval()
    with torch.no_grad():
        dataiter = iter(data_loader)
        img, _ = next(dataiter)
        img_flat = img.view(img.size(0), -1).to(device)
        output = model(img_flat)
        
        # Reshape back to images
        img = img.cpu().numpy()
        output = output.view(output.size(0), 1, 28, 28).cpu().numpy()
        
        # Plot results
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))
        
        for i in range(num_samples):
            # Original images (top row)
            axes[0, i].imshow(img[i].squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed images (bottom row)
            axes[1, i].imshow(output[i].squeeze(), cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.suptitle('Autoencoder Reconstructions', fontsize=16)
        plt.tight_layout()
        plt.savefig('autoencoder_reconstructions.png', dpi=300, bbox_inches='tight')
        plt.show()

class VariationalAutoencoder(nn.Module):
    """Convolutional Variational Autoencoder for image generation"""
    
    def __init__(self, 
                 in_channels: int = 3, 
                 base_channels: int = 32, 
                 latent_dim: int = 1024):
        """
        Initialize the VAE architecture.
        
        Args:
            in_channels: Number of input channels
            base_channels: Base number of channels for convolutional layers
            latent_dim: Dimension of latent space
        """
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # Latent space layers
        self.fc_mu = nn.Linear(base_channels * 8 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(base_channels * 8 * 4 * 4, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, base_channels * 8 * 4 * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(base_channels, in_channels, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input into latent distribution parameters"""
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector into reconstructed image"""
        x = self.decoder_input(z)
        x = x.view(-1, 256, 4, 4)  # Reshape to match encoder output
        return self.decoder(x)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x: torch.Tensor, 
            x: torch.Tensor, 
            mu: torch.Tensor, 
            logvar: torch.Tensor, 
            reconstruction_weight: float = 1.0,
            kl_weight: float = 1.0) -> torch.Tensor:
    """
    VAE loss function combining reconstruction loss and KL divergence.
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        reconstruction_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence
        
    Returns:
        Total loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return reconstruction_weight * recon_loss + kl_weight * kl_loss

def prepare_face_data(data_path: str, batch_size: int = 32) -> DataLoader:
    """
    Prepare face image dataset for VAE training.
    
    Args:
        data_path: Path to face images
        batch_size: Batch size for training
        
    Returns:
        DataLoader for face dataset
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_vae(model: nn.Module, 
             data_loader: DataLoader, 
             epochs: int = 20, 
             lr: float = 0.001) -> List[float]:
    """
    Train a Variational Autoencoder.
    
    Args:
        model: VAE model to train
        data_loader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        List of training losses per epoch
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    print("Training Variational Autoencoder...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1:2d}/{epochs}] | Loss: {avg_loss:.6f}')
    
    return losses

def generate_samples(model: nn.Module, 
                    num_samples: int = 25, 
                    latent_dim: int = 1024) -> torch.Tensor:
    """
    Generate new samples from trained VAE.
    
    Args:
        model: Trained VAE model
        num_samples: Number of samples to generate
        latent_dim: Dimension of latent space
        
    Returns:
        Generated images
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decode(z)
    return samples.cpu()

def visualize_generated_samples(samples: torch.Tensor, 
                               grid_size: Tuple[int, int] = (5, 5)) -> None:
    """
    Visualize generated samples from VAE.
    
    Args:
        samples: Generated images tensor
        grid_size: Grid size for visualization
    """
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(samples):
            img = samples[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')
    
    plt.suptitle('VAE Generated Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig('vae_generated_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run both autoencoder experiments"""
    
    print("=" * 60)
    print("AUTOENCODERS COMPARISON EXPERIMENT")
    print("=" * 60)
    
    # Part 1: Fully Connected Autoencoder for MNIST
    print("\n1. Training Fully Connected Autoencoder on MNIST")
    print("-" * 50)
    
    mnist_loader = prepare_mnist_data(batch_size=100, noise_std=0.1)
    fc_autoencoder = FullyConnectedAutoencoder(latent_dim=3)
    
    # Train autoencoder
    fc_losses = train_autoencoder(fc_autoencoder, mnist_loader, epochs=10, lr=0.001)
    
    # Visualize reconstructions
    visualize_reconstructions(fc_autoencoder, mnist_loader, num_samples=10)
    
    # Part 2: Variational Autoencoder for Face Generation
    print("\n2. Training Variational Autoencoder on Face Images")
    print("-" * 50)
    
    # Extract face data if needed
    face_data_path = "faces_data"
    if not os.path.exists(face_data_path):
        print("Please extract faces.zip to faces_data/ directory")
        return
    
    # Prepare face data
    face_loader = prepare_face_data(face_data_path, batch_size=32)
    
    # Create and train VAE
    vae = VariationalAutoencoder(latent_dim=1024)
    vae_losses = train_vae(vae, face_loader, epochs=20, lr=0.001)
    
    # Generate and visualize samples
    generated_samples = generate_samples(vae, num_samples=25)
    visualize_generated_samples(generated_samples)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fc_losses, label='FC Autoencoder', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FC Autoencoder Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(vae_losses, label='VAE', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()

