# NLP and Computer Vision Projects Collection

A collection of projects on natural language processing and computer vision using deep learning techniques.

## 📁 Projects

### 1. Coronavirus Sentiment Analysis
**File:** `coronavirus_sentiment_analysis.py`

**Description:** 
Sentiment analysis of COVID-19 tweets using various NLP approaches:
- Traditional methods (Bag-of-Words, TF-IDF) with Logistic Regression
- Deep learning with BERT transformer model

**Key Features:**
- Text preprocessing (cleaning, stemming, lemmatization)
- Handling class imbalance
- Comparison of traditional vs. modern approaches
- Results visualization

**Technologies:** 
- Python, PyTorch, Transformers
- Scikit-learn, NLTK
- BERT (bert-base-uncased)

---

### 2. Autoencoders Comparison
**File:** `autoencoders_comparison.py`

**Description:** 
Comparison of Fully Connected Autoencoder and Variational Autoencoder:
- Denoising autoencoder for MNIST with noise
- Variational autoencoder for face generation

**Key Features:**
- Implementation of two autoencoder types
- Reconstruction and generation visualization
- Training on noisy data

**Technologies:**
- PyTorch
- MNIST dataset
- Convolutional networks

---

### 3. MNIST Classification with PyTorch
**File:** `mnist_classification_pytorch.py`

**Description:** 
Handwritten digit classification on MNIST dataset using:
- Fully connected neural networks
- PyTorch framework

**Key Features:**
- MNIST data preprocessing
- Neural network implementation with PyTorch
- Training process visualization
- Model accuracy evaluation

**Technologies:**
- PyTorch
- MNIST dataset
- Fully connected neural networks

---

### 4. Transfer Learning Comparison
**File:** `transfer_learning_comparison.py`

**Description:** 
Comparison of feature extraction vs. fine-tuning approaches for:
- AlexNet
- VGG16
- ResNet18

**Key Features:**
- Comparison of two transfer learning strategies
- Analysis of different architecture effectiveness
- Training results visualization

**Technologies:**
- PyTorch, Torchvision
- Pretrained models (AlexNet, VGG16, ResNet18)
- Transfer learning techniques

---

### 5. LeNet Architectures Comparison on CIFAR-10
**File:** `cifar10_lenet_architectures_comparison.py`

**Description:** 
Comparison of various LeNet architecture modifications:
- Original LeNet with Tanh
- Modifications with ReLU and MaxPooling
- Modern version with additional layers
- Version with batch normalization

**Key Features:**
- Activation function comparison
- Pooling methods analysis
- Batch normalization impact evaluation

**Technologies:**
- PyTorch, Torchvision
- CIFAR-10 dataset
- Convolutional neural networks

---

### 6. Optimizers Comparison on FashionMNIST
**File:** `fashion_mnist_optimizers_comparison.py`

**Description:** 
Comparison of different optimizers:
- SGD with momentum
- AdaGrad, RMSProp
- Adam, AdaDelta, Adamax

**Key Features:**
- Learning curves comparison
- Convergence speed analysis
- Optimizers effectiveness evaluation

**Technologies:**
- PyTorch
- FashionMNIST dataset
- Various optimizers

---

## 🛠 Installation and Setup

### Requirements
```bash
pip install torch torchvision transformers
pip install scikit-learn nltk matplotlib seaborn pandas numpy
