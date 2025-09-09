import torch                 # tensor library (the core math library)
import torch.nn as nn        # neural network building blocks (layers that hold weights & biases)
import torch.nn.functional as F  # stateless ops (activations like ReLU that don’t have weights)

class MLP(nn.Module):  # MLP = Multi-Layer Perceptron
    def __init__(self):
        super().__init__()   # init parent class

        # Layers
        self.fc1 = nn.Linear(28*28, 256)  # first fully connected layer
        self.fc2 = nn.Linear(256, 10)     # second fully connected layer

    def forward(self, x):
        # Flatten 28x28 image → 784 vector
        x = x.view(x.size(0), -1)

        # Layer 1 + ReLU activation
        x = F.relu(self.fc1(x))

        # Layer 2 (logits, not softmax yet)
        x = self.fc2(x)
        return x