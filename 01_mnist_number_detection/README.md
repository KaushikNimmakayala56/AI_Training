# Number Recognition with MNIST

A PyTorch-based neural network project for recognizing handwritten digits from the MNIST dataset.

## Overview

This project implements a Multi-Layer Perceptron (MLP) neural network to classify handwritten digits (0-9) using the famous MNIST dataset. The model achieves high accuracy in digit recognition through supervised learning.

## Features

- **Simple MLP Architecture**: Two-layer fully connected neural network
- **MNIST Dataset**: Uses the standard MNIST handwritten digit dataset
- **PyTorch Implementation**: Built with PyTorch for efficient training and inference
- **Data Visualization**: Includes tools to explore and visualize the dataset
- **Model Persistence**: Saves trained models for future use

## Project Structure

```
01_mnist_number_detection/
├── data/                    # MNIST dataset storage
│   └── MNIST/
│       └── raw/            # Raw MNIST data files
├── model.py                # Neural network model definition
├── train.py                # Training script
├── explore_data            # Data exploration and visualization
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KaushikNimmakayala56/AI_Training.git
cd AI_Training/01_mnist_number_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script to train the neural network:

```bash
python train.py
```

This will:
- Download the MNIST dataset (if not already present)
- Train the MLP model for 3 epochs
- Display training loss and test accuracy
- Save the trained model as `mnist_mlp.pt`

### Exploring the Data

To visualize sample images from the MNIST dataset:

```bash
python explore_data
```

This script will display a sample handwritten digit with its corresponding label.

## Model Architecture

The neural network consists of:

- **Input Layer**: 784 neurons (28×28 flattened image)
- **Hidden Layer**: 256 neurons with ReLU activation
- **Output Layer**: 10 neurons (one for each digit 0-9)

```
Input (784) → Hidden (256) → Output (10)
```

## Dependencies

- `torch`: PyTorch deep learning framework
- `torchvision`: Computer vision utilities for PyTorch
- `matplotlib`: Data visualization library

## Training Details

- **Optimizer**: Adam with learning rate 1e-3
- **Loss Function**: Cross Entropy Loss
- **Batch Size**: 64 for training, 256 for testing
- **Epochs**: 3 (configurable in train.py)
- **Device**: CPU (suitable for MNIST dataset size)

## Expected Performance

The model typically achieves:
- Training loss reduction over epochs
- Test accuracy of 95%+ after just 3 epochs

## Future Enhancements

Potential improvements could include:
- Convolutional Neural Network (CNN) architecture
- Data augmentation techniques
- Hyperparameter tuning
- Model evaluation metrics and visualization
- Real-time digit recognition from camera input

## License

This project is open source and available under the MIT License.
