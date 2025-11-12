# MNIST Handwritten Digit Classifier

A neural network implementation for classifying handwritten digits from the MNIST dataset, developed as part of the Udacity Advanced Machine Learning Program.

This project was completed as part of the **Udacity Advanced AWS Machine Learning Fundamentals Nanodegree Program**.


> #### Reviewer Note for my Project Submission on Udacity Platform
> *"Well done on your work! You showed initiative by experimenting with dropout, which directly improved your model’s accuracy and helped you meet the passing requirements. This demonstrates a solid understanding of how hyperparameter tuning can impact performance. Keep building on this by trying out additional hyperparameters in future experiments. Great progress — keep it up! 🚀"*
---


## Overview

This project implements a Multi-Layer Perceptron (MLP) using PyTorch to classify handwritten digits (0-9) from the MNIST dataset with **97.23% test accuracy**.

## Model Architecture

- **Input Layer**: 784 neurons (28×28 flattened images)
- **Hidden Layer 1**: 125 neurons with ReLU activation
- **Hidden Layer 2**: 70 neurons with ReLU activation
- **Dropout**: 30% dropout rate for regularization
- **Output Layer**: 10 neurons (one per digit class)

## Key Features

- Data preprocessing with normalization and flattening
- Adam optimizer with learning rate of 0.001
- Cross-entropy loss function
- Training over 10 epochs
- Dropout regularization to prevent overfitting

## Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 98.10% |
| Test Accuracy | 97.23% |
| Test Loss | 0.0981 |

## Requirements
```
torch
torchvision
matplotlib
numpy
```

## Dataset

The MNIST dataset contains:
- 60,000 training images
- 10,000 test images
- 28×28 grayscale images of handwritten digits

## Training Process

1. Data loading and preprocessing
2. Model initialization
3. Training with batch size of 32
4. Performance monitoring with loss and accuracy tracking
5. Model evaluation on test set

## Model Improvements

Two model versions were implemented:
- **Version 1**: Basic MLP → 96.98% test accuracy
- **Version 2**: MLP with dropout → **97.23% test accuracy**

The addition of 30% dropout in the second hidden layer improved generalization and reduced overfitting.

## Usage
```python
# Load the trained model
model = MLP_net2()
model.load_state_dict(torch.load('best_mnist_model.pth'))
model.eval()
```

