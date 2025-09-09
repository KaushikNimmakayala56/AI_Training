# MNIST Number Detection Neural Network

This is Project #1 in the AI Training repository - a simple neural network for handwritten digit recognition using the MNIST dataset.

## Overview

This project implements a Multi-Layer Perceptron (MLP) neural network to classify handwritten digits (0-9) from the famous MNIST dataset. The model uses PyTorch and achieves good accuracy on the test set.

## Files

- `model.py` - Contains the MLP neural network architecture
- `train.py` - Training script that loads data, trains the model, and evaluates performance
- `requirements.txt` - Python dependencies
- `mnist_mlp.pt` - Trained model weights (generated after training)

## Model Architecture

- **Input**: 28x28 grayscale images (flattened to 784 features)
- **Hidden Layer**: 256 neurons with ReLU activation
- **Output Layer**: 10 neurons (one for each digit 0-9)
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam with learning rate 1e-3

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run training:
   ```bash
   python train.py
   ```

The script will:
- Download the MNIST dataset automatically
- Train the model for 3 epochs
- Display training loss and test accuracy
- Save the trained model as `mnist_mlp.pt`

## Expected Results

The model typically achieves:
- Training loss: ~0.1-0.2 after 3 epochs
- Test accuracy: ~95-97%

## Next Steps

This is a basic implementation. Future improvements could include:
- Convolutional Neural Networks (CNNs)
- Data augmentation
- Hyperparameter tuning
- More sophisticated architectures
