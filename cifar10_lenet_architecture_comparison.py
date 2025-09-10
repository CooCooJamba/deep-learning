"""
CIFAR-10 LeNet Architecture Comparison with PyTorch

This script compares different variations of LeNet architecture on the CIFAR-10 dataset
using PyTorch. It evaluates original LeNet with Tanh, ReLU activation, MaxPooling,
and a modernized version with additional convolutional layers.

Author: VShulgin
Date: 2022-08-10
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import time

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

def load_cifar10_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """
    Load and preprocess CIFAR-10 dataset.
    
    Args:
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Define transformations with normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    data_train = CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    data_test = CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        data_train, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        data_test, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader

def visualize_sample_images(data_loader: DataLoader, 
                          class_names: List[str], 
                          num_images: int = 9) -> None:
    """
    Visualize sample images from the dataset.
    
    Args:
        data_loader: DataLoader containing images
        class_names: List of class names
        num_images: Number of images to display
    """
    images, labels = next(iter(data_loader))
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < num_images:
            # Convert tensor to numpy and denormalize
            img = images[i].numpy().transpose(1, 2, 0)
            img = np.clip(img * np.array([0.2023, 0.1994, 0.2010]) + 
                         np.array([0.4914, 0.4822, 0.4465]), 0, 1)
            
            ax.imshow(img)
            ax.set_title(f'Class: {class_names[labels[i].item()]}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_lenet_old() -> nn.Sequential:
    """Create original LeNet architecture with Tanh activation and AvgPool"""
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
        nn.Tanh(),
        nn.Flatten(),
        nn.Linear(in_features=120, out_features=84),
        nn.Tanh(),
        nn.Linear(in_features=84, out_features=10)
    )

def create_lenet_relu() -> nn.Sequential:
    """Create LeNet with ReLU activation instead of Tanh"""
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=120, out_features=84),
        nn.ReLU(),
        nn.Linear(in_features=84, out_features=10)
    )

def create_lenet_maxpool() -> nn.Sequential:
    """Create LeNet with ReLU activation and MaxPooling"""
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=120, out_features=84),
        nn.ReLU(),
        nn.Linear(in_features=84, out_features=10)
    )

def create_lenet_modern() -> nn.Sequential:
    """Create modernized LeNet with additional convolutional layers"""
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=120, out_features=84),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=84, out_features=10)
    )

def train_model(model: nn.Module, 
               train_loader: DataLoader, 
               test_loader: DataLoader, 
               num_epochs: int = 10,
               model_name: str = "Model") -> Tuple[List[float], List[float]]:
    """
    Train a model and return training history.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        model_name: Name of the model for printing
        
    Returns:
        Tuple of (train_losses, test_losses, test_accuracies)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100 * correct / total
        
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Test Loss: {test_loss:.4f}, '
                  f'Accuracy: {accuracy:.2f}%')
    
    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    
    return train_losses, test_losses, test_accuracies

def plot_results(loss_dict: Dict[str, Tuple[List[float], List[float], List[float]]]) -> None:
    """
    Plot training and test results for all models.
    
    Args:
        loss_dict: Dictionary with model names as keys and 
                  (train_losses, test_losses, test_accuracies) as values
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot training loss
    for model_name, (train_losses, test_losses, test_accuracies) in loss_dict.items():
        axes[0].plot(train_losses, label=model_name, linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot test loss
    for model_name, (train_losses, test_losses, test_accuracies) in loss_dict.items():
        axes[1].plot(test_losses, label=model_name, linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test Loss")
    axes[1].set_title("Test Loss Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot test accuracy
    for model_name, (train_losses, test_losses, test_accuracies) in loss_dict.items():
        axes[2].plot(test_accuracies, label=model_name, linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Test Accuracy (%)")
    axes[2].set_title("Test Accuracy Comparison")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lenet_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the LeNet architecture comparison."""
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10_data(batch_size=64)
    
    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Visualize sample images
    print("Visualizing sample images...")
    visualize_sample_images(train_loader, class_names)
    
    # Create and train different LeNet variants
    models = {
        "LeNet_Original": create_lenet_old(),
        "LeNet_ReLU": create_lenet_relu(),
        "LeNet_MaxPool": create_lenet_maxpool(),
        "LeNet_Modern": create_lenet_modern()
    }
    
    results = {}
    
    for model_name, model in models.items():
        train_losses, test_losses, test_accuracies = train_model(
            model, train_loader, test_loader, 
            num_epochs=30, model_name=model_name
        )
        results[model_name] = (train_losses, test_losses, test_accuracies)
    
    # Plot results
    print("Plotting comparison results...")
    plot_results(results)
    
    # Print final results
    print("\nFinal Results Comparison:")
    print("-" * 50)
    for model_name, (train_losses, test_losses, test_accuracies) in results.items():
        print(f"{model_name:<20}: Test Loss = {test_losses[-1]:.4f}, "
              f"Accuracy = {test_accuracies[-1]:.2f}%")

if __name__ == "__main__":
    main()

