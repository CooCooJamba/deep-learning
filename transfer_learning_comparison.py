"""
Transfer Learning Comparison: AlexNet, VGGNet, and ResNet on Custom Dataset

This script compares feature extraction vs fine-tuning approaches for three popular
CNN architectures (AlexNet, VGG16, ResNet18) on a custom image classification task.

Author: VShulgin
Date: 2022-08-13
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Tuple
import time
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

def prepare_data(data_path: str = 'images', 
                batch_size: int = 32,
                train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Prepare and load image data with appropriate transformations.
    
    Args:
        data_path: Path to the image dataset
        batch_size: Batch size for data loaders
        train_ratio: Ratio of training data
        
    Returns:
        Tuple of (train_loader, test_loader, class_names)
    """
    # Data augmentation for training, simple transformation for testing
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    train_dataset = ImageFolder(data_path, transform=train_transform)
    test_dataset = ImageFolder(data_path, transform=test_transform)
    
    # Split dataset
    train_size = int(train_ratio * len(train_dataset))
    test_size = len(train_dataset) - train_size
    
    train_subset, test_subset = random_split(
        train_dataset, [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_subset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.classes

def visualize_samples(data_loader: DataLoader, 
                     class_names: List[str], 
                     num_samples: int = 9) -> None:
    """
    Visualize sample images from the dataset.
    
    Args:
        data_loader: DataLoader containing images
        class_names: List of class names
        num_samples: Number of samples to display
    """
    images, labels = next(iter(data_loader))
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < num_samples:
            # Denormalize image
            img = images[i].numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.set_title(f'Class: {class_names[labels[i]]}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_model(model_name: str, 
                num_classes: int, 
                feature_extraction: bool = True) -> Tuple[nn.Module, List]:
    """
    Create a pretrained model with modified classifier.
    
    Args:
        model_name: Name of the model ('alexnet', 'vgg16', 'resnet18')
        num_classes: Number of output classes
        feature_extraction: Whether to freeze feature extractor weights
        
    Returns:
        Tuple of (model, parameters_to_train)
    """
    if model_name.lower() == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        if feature_extraction:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        params_to_train = model.classifier[6].parameters()
        
    elif model_name.lower() == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        if feature_extraction:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        params_to_train = model.classifier[6].parameters()
        
    elif model_name.lower() == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if feature_extraction:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        params_to_train = model.fc.parameters()
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model, list(params_to_train)

def train_model(model: nn.Module, 
               train_loader: DataLoader, 
               test_loader: DataLoader, 
               criterion: nn.Module,
               optimizer: optim.Optimizer,
               scheduler: optim.lr_scheduler._LRScheduler,
               epochs: int = 10,
               model_name: str = "Model") -> Tuple[List[float], List[float], List[float]]:
    """
    Train a model and return training history.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epochs: Number of training epochs
        model_name: Name of the model for printing
        
    Returns:
        Tuple of (train_losses, test_losses, test_accuracies)
    """
    model = model.to(device)
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    print(f"\nTraining {model_name}...")
    print("-" * 60)
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1:2d}/{epochs}] | '
              f'Train Loss: {train_loss:.4f} | '
              f'Test Loss: {test_loss:.4f} | '
              f'Test Acc: {test_accuracy:.2f}%')
    
    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    
    return train_losses, test_losses, test_accuracies

def plot_comparison(results: Dict[str, Dict[str, Tuple[List[float], List[float], List[float]]]]) -> None:
    """
    Plot comparison results between different models and approaches.
    
    Args:
        results: Dictionary containing training results
    """
    # Plot feature extraction results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature Extraction - Loss
    for model_name, metrics in results.items():
        train_loss, test_loss, test_acc = metrics["Feature Extraction"]
        axes[0, 0].plot(train_loss, label=f'{model_name}', linewidth=2)
    axes[0, 0].set_title('Feature Extraction - Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Feature Extraction - Accuracy
    for model_name, metrics in results.items():
        train_loss, test_loss, test_acc = metrics["Feature Extraction"]
        axes[0, 1].plot(test_acc, label=f'{model_name}', linewidth=2)
    axes[0, 1].set_title('Feature Extraction - Test Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Fine-tuning - Loss
    for model_name, metrics in results.items():
        train_loss, test_loss, test_acc = metrics["Fine-tuning"]
        axes[1, 0].plot(train_loss, label=f'{model_name}', linewidth=2)
    axes[1, 0].set_title('Fine-tuning - Training Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Fine-tuning - Accuracy
    for model_name, metrics in results.items():
        train_loss, test_loss, test_acc = metrics["Fine-tuning"]
        axes[1, 1].plot(test_acc, label=f'{model_name}', linewidth=2)
    axes[1, 1].set_title('Fine-tuning - Test Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transfer_learning_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Individual model comparisons
    for model_name in results.keys():
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Training loss comparison
        fe_train_loss, fe_test_loss, fe_test_acc = results[model_name]["Feature Extraction"]
        ft_train_loss, ft_test_loss, ft_test_acc = results[model_name]["Fine-tuning"]
        
        axes[0].plot(fe_train_loss, label='Feature Extraction', linewidth=2)
        axes[0].plot(ft_train_loss, label='Fine-tuning', linewidth=2)
        axes[0].set_title(f'{model_name} - Training Loss Comparison')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Test accuracy comparison
        axes[1].plot(fe_test_acc, label='Feature Extraction', linewidth=2)
        axes[1].plot(ft_test_acc, label='Fine-tuning', linewidth=2)
        axes[1].set_title(f'{model_name} - Test Accuracy Comparison')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run the transfer learning comparison."""
    # Prepare data
    print("Loading and preparing data...")
    train_loader, test_loader, class_names = prepare_data(
        data_path='images', 
        batch_size=32,
        train_ratio=0.8
    )
    
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    # Visualize samples
    print("Visualizing sample images...")
    visualize_samples(train_loader, class_names)
    
    # Initialize results dictionary
    results = {
        "AlexNet": {},
        "VGG16": {},
        "ResNet18": {}
    }
    
    # Training parameters
    epochs = 15
    criterion = nn.CrossEntropyLoss()
    
    # Test both approaches for each model
    models_to_test = ["AlexNet", "VGG16", "ResNet18"]
    approaches = ["Feature Extraction", "Fine-tuning"]
    
    for model_name in models_to_test:
        for approach in approaches:
            feature_extraction = (approach == "Feature Extraction")
            
            print(f"\n{approach} with {model_name}")
            print("=" * 50)
            
            # Create model
            model, params_to_train = create_model(
                model_name, 
                len(class_names), 
                feature_extraction=feature_extraction
            )
            
            # Set up optimizer and scheduler
            if feature_extraction:
                optimizer = optim.Adam(params_to_train, lr=0.001, weight_decay=1e-4)
            else:
                optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
            
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            
            # Train model
            train_losses, test_losses, test_accuracies = train_model(
                model, train_loader, test_loader, criterion,
                optimizer, scheduler, epochs=epochs,
                model_name=f"{model_name} ({approach})"
            )
            
            results[model_name][approach] = (train_losses, test_losses, test_accuracies)
    
    # Plot results
    print("\nPlotting comparison results...")
    plot_comparison(results)
    
    # Print final results
    print("\nFinal Results Comparison:")
    print("=" * 60)
    for model_name in results.keys():
        for approach in approaches:
            train_loss, test_loss, test_acc = results[model_name][approach]
            print(f"{model_name:10} {approach:20}: "
                  f"Final Test Acc = {test_acc[-1]:6.2f}% | "
                  f"Final Loss = {test_loss[-1]:.4f}")

if __name__ == "__main__":
    main()

