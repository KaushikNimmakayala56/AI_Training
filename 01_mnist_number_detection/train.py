import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP

# 1) Data: MNIST tensors in [0,1]
transform = transforms.ToTensor()
train_ds = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

# 2) Model / loss / optimizer
device = torch.device("cpu")  # CUDA not available on your Mac; CPU is fine for MNIST
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3) Train
def train_one_epoch(epoch):
    model.train()
    running = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)            # forward
        loss = criterion(logits, y)  # compute loss
        loss.backward()              # backprop (compute grads)
        optimizer.step()             # update weights
        running += loss.item()
    avg = running / len(train_loader)
    print(f"Epoch {epoch}: train loss = {avg:.4f}")

# 4) Evaluate (accuracy)
@torch.no_grad()
def evaluate():
    model.eval()
    correct, total = 0, 0
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    acc = 100.0 * correct / total
    print(f"Test accuracy: {acc:.2f}%")
    return acc

# 5) Run a few epochs
for ep in range(1, 4):  # 3 epochs to start
    train_one_epoch(ep)
    evaluate()

# 6) Save model
torch.save(model.state_dict(), "mnist_mlp.pt")
print("Saved to mnist_mlp.pt")