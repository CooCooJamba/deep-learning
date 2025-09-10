#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
FashionMNIST Optimizers Comparison with PyTorch

This script compares different optimization algorithms on the FashionMNIST dataset
using PyTorch. It evaluates SGD with momentum, AdaGrad, RMSProp, Adam, AdaDelta,
and Adamax optimizers.

Author: VShulgin
Date: 2022-07-28
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_fashion_mnist_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and preprocess FashionMNIST dataset.
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) tensors
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Extract data and labels
    x_train = train_dataset.data.float() / 255.0
    y_train = train_dataset.targets
    x_test = test_dataset.data.float() / 255.0
    y_test = test_dataset.targets
    
    # Flatten images
    x_train = x_train.view(-1, 28 * 28)
    x_test = x_test.view(-1, 28 * 28)
    
    return x_train, y_train, x_test, y_test

def visualize_sample_images(x_data: torch.Tensor, y_data: torch.Tensor, 
                          class_names: List[str], num_images: int = 9) -> None:
    """
    Visualize sample images from the dataset.
    
    Args:
        x_data: Image data tensor
        y_data: Labels tensor
        class_names: List of class names
        num_images: Number of images to display
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < num_images:
            img = x_data[i].view(28, 28).numpy()
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Class: {class_names[y_data[i].item()]}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

class FashionMNISTNet(nn.Module):
    """
    Neural network for FashionMNIST classification.
    """
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, 
                 output_size: int = 10) -> None:
        """
        Initialize the network architecture.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layer
            output_size: Number of output classes
        """
        super(FashionMNISTNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.network(x)

def test_optimizer(net: nn.Module, optimizer: torch.optim.Optimizer, 
                  x_train: torch.Tensor, y_train: torch.Tensor,
                  x_test: torch.Tensor, y_test: torch.Tensor,
                  batch_size: int = 100, epochs: int = 50) -> Tuple[List[float], List[float]]:
    """
    Test an optimizer on the FashionMNIST dataset.
    
    Args:
        net: Neural network model
        optimizer: Optimizer to test
        x_train: Training data
        y_train: Training labels
        x_test: Test data
        y_test: Test labels
        batch_size: Batch size for training
        epochs: Number of training epochs
        
    Returns:
        Tuple of (train_losses, test_losses)
    """
    criterion = nn.CrossEntropyLoss()
    loss_values_train = []
    loss_values_test = []
    
    for epoch in range(epochs):
        # Training phase
        net.train()
        order = np.random.permutation(len(x_train))
        
        for start_index in range(0, len(x_train), batch_size):
            batch_index = order[start_index:start_index + batch_size]
            x_batch = x_train[batch_index]
            y_batch = y_train[batch_index]
            
            optimizer.zero_grad()
            y_preds = net(x_batch)
            loss_val = criterion(y_preds, y_batch)
            loss_val.backward()
            optimizer.step()
        
        # Evaluation phase
        net.eval()
        with torch.no_grad():
            # Training loss
            y_pred_train = net(x_train)
            loss_train = criterion(y_pred_train, y_train)
            loss_values_train.append(loss_train.item())
            
            # Test loss
            y_pred_test = net(x_test)
            loss_test = criterion(y_pred_test, y_test)
            loss_values_test.append(loss_test.item())
    
    return loss_values_train, loss_values_test

def plot_loss_curves(loss_dict: Dict[str, Tuple[List[float], List[float]]], 
                    title_suffix: str) -> None:
    """
    Plot training and test loss curves for different optimizers.
    
    Args:
        loss_dict: Dictionary with optimizer names as keys and (train_losses, test_losses) as values
        title_suffix: Suffix for plot titles
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    for optim_name, (train_losses, test_losses) in loss_dict.items():
        plt.plot(train_losses, label=optim_name, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Training Loss - {title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot test loss
    plt.subplot(1, 2, 2)
    for optim_name, (train_losses, test_losses) in loss_dict.items():
        plt.plot(test_losses, label=optim_name, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Test Loss - {title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the optimizer comparison."""
    # Load and preprocess data
    print("Loading FashionMNIST dataset...")
    x_train, y_train, x_test, y_test = load_fashion_mnist_data()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Class names for FashionMNIST
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    # Visualize sample images
    print("Visualizing sample images...")
    visualize_sample_images(x_train.view(-1, 28, 28), y_train, class_names)
    
    # Test different optimizers
    loss_optim = {}
    optimizers_config = {
        "SGD with Momentum": {
            "optimizer": lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
            "hidden_size": 128
        },
        "AdaGrad": {
            "optimizer": lambda params: torch.optim.Adagrad(params, lr=0.01),
            "hidden_size": 128
        },
        "RMSProp": {
            "optimizer": lambda params: torch.optim.RMSprop(params, lr=0.001, weight_decay=1e-5),
            "hidden_size": 128
        },
        "Adam": {
            "optimizer": lambda params: torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999)),
            "hidden_size": 128
        },
        "AdaDelta": {
            "optimizer": lambda params: torch.optim.Adadelta(params, lr=1.0, rho=0.9),
            "hidden_size": 128
        },
        "Adamax": {
            "optimizer": lambda params: torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999)),
            "hidden_size": 128
        }
    }
    
    print("Testing different optimizers...")
    for optim_name, config in optimizers_config.items():
        print(f"Testing {optim_name}...")
        net = FashionMNISTNet(
            input_size=784, 
            hidden_size=config["hidden_size"], 
            output_size=10
        )
        
        optimizer = config["optimizer"](net.parameters())
        train_losses, test_losses = test_optimizer(
            net, optimizer, x_train, y_train, x_test, y_test,
            batch_size=100, epochs=50
        )
        
        loss_optim[optim_name] = (train_losses, test_losses)
    
    # Plot results
    print("Plotting results...")
    plot_loss_curves(loss_optim, "FashionMNIST Optimizers Comparison")
    
    # Print final losses
    print("\nFinal Test Losses:")
    for optim_name, (train_losses, test_losses) in loss_optim.items():
        print(f"{optim_name}: {test_losses[-1]:.4f}")

if __name__ == "__main__":
    main()

