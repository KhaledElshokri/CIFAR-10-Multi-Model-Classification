import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 512),  # First hidden layer
            nn.ReLU(),
            nn.BatchNorm1d(512),  # Batch normalization
            nn.Linear(512, 512),  # Second hidden layer
            nn.ReLU(),
            nn.Linear(512, 10)    # Output layer
        )

    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation function with metrics calculation
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Calculate predictions and accuracy
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            
            # Save predictions and labels for metrics calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Metrics calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("Confusion Matrix:\n", conf_matrix)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save confusion matrix to a CSV file
    csv_filename = "confusion_matrix_mlp.csv"
    np.savetxt(csv_filename, conf_matrix, delimiter=",", fmt='%d')
    print(f"Confusion matrix saved to {csv_filename}")
    
    # Return metrics
    accuracy_final = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy_final

if __name__ == "__main__":
    # Load the preprocessed feature data and labels
    train_features = np.load("train_features_full.npy")
    train_labels = np.load("train_labels.npy")
    test_features = np.load("test_features_full.npy")
    test_labels = np.load("test_labels.npy")

    # Convert to PyTorch tensors
    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Initialize the MLP model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    epochs = 50
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Calculate and display the overall accuracy after training
    print("\nCalculating overall accuracy on the test set...")
    final_test_loss, final_test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Final Test Loss: {final_test_loss:.4f}, Final Overall Test Accuracy: {final_test_accuracy:.4f}")
