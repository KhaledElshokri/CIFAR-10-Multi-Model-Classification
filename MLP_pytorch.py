import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x

# Load the data
train_features = np.load("train_features_full.npy")
test_features = np.load("test_features_full.npy")
train_labels = np.load("train_labels.npy")
test_labels = np.load("test_labels.npy")

# Convert data to PyTorch tensors
train_features = torch.tensor(train_features, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_features = torch.tensor(test_features, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Initialize model, loss function, and optimizer
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
num_epochs = 40
batch_size = 32
for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(train_features.size(0))
    for i in range(0, train_features.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = train_features[indices], train_labels[indices]

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(test_features)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
