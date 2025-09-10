"""
MNIST Classification with PyTorch

This script implements a neural network for classifying MNIST handwritten digits
using PyTorch. It includes data loading, preprocessing, model definition,
training, and evaluation.

Author: VShulgin
Date: 2022-07-17
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Set random seeds for reproducibility
def set_seed(seed=0):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Download and load MNIST dataset
def load_mnist_data():
    """Load and preprocess MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    return train_dataset, test_dataset

# Define the neural network architecture
class MNISTNet(nn.Module):
    """Neural network for MNIST classification"""
    
    def __init__(self, hidden_size=128):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.Tanh()
        self.out = nn.Linear(hidden_size, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.out(x)
        return x
    
    def predict(self, x):
        """Make predictions with the model"""
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            outputs = self.forward(x)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu()

# Training function
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    """Train the model and return training history"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    train_losses = []
    test_losses = []
    accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
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
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100 * correct / total
        
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Test Loss: {test_loss:.4f}, '
                  f'Accuracy: {accuracy:.2f}%')
    
    return train_losses, test_losses, accuracies

# Main execution
if __name__ == "__main__":
    # Load data
    train_dataset, test_dataset = load_mnist_data()
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    
    # Initialize model
    model = MNISTNet(hidden_size=128).to(device)
    print(f"Model architecture:\n{model}")
    
    # Train model
    print("Starting training...")
    train_losses, test_losses, accuracies = train_model(
        model, train_loader, test_loader, epochs=40, lr=0.001
    )
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model.predict(images)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    final_accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {final_accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    # Save model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model saved as 'mnist_model.pth'")

