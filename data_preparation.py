# data_preparation.py

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from sklearn.decomposition import PCA
import numpy as np
import os

def load_data():
    # Define transformations (resizing and normalization for ResNet-18)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Select 500 training and 100 test images per class
    train_subset_indices = []
    test_subset_indices = []
    train_labels = []
    test_labels = []

    class_counts = {i: 0 for i in range(10)}
    for i, (_, label) in enumerate(train_dataset):
        if class_counts[label] < 500:
            train_subset_indices.append(i)
            train_labels.append(label)
            class_counts[label] += 1

    class_counts = {i: 0 for i in range(10)}
    for i, (_, label) in enumerate(test_dataset):
        if class_counts[label] < 100:
            test_subset_indices.append(i)
            test_labels.append(label)
            class_counts[label] += 1

    print("Creating data loaders...")
    train_subset = Subset(train_dataset, train_subset_indices)
    test_subset = Subset(test_dataset, test_subset_indices)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    # Debugging print statements
    print("Number of training labels:", len(train_labels))
    print("Number of test labels:", len(test_labels))

    return train_loader, test_loader, np.array(train_labels), np.array(test_labels)

def extract_features(loader):
    print("Extracting features using ResNet-18...")
    # Load pre-trained ResNet-18 with specified weights and modify to remove the final layer
    resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18 = resnet18.to(device)
    resnet18.eval()
    
    features = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            output = resnet18(images)
            features.append(output.squeeze().cpu().numpy())
    return np.vstack(features)

def reduce_dimensions(train_features, test_features):
    print("Applying PCA for dimensionality reduction...")
    # Apply PCA to reduce dimensions from 512 to 50
    pca = PCA(n_components=100)
    train_features_pca = pca.fit_transform(train_features)
    test_features_pca = pca.transform(test_features)
    return train_features_pca, test_features_pca

if __name__ == "__main__":
    print("Starting data preparation...")
    train_loader, test_loader, train_labels, test_labels = load_data()
    train_features = extract_features(train_loader)
    test_features = extract_features(test_loader)
    train_features_pca, test_features_pca = reduce_dimensions(train_features, test_features)
    
    # Save the PCA features and labels for use in model training
    print("Saving PCA features and labels...")
    np.save("train_features_pca.npy", train_features_pca)
    np.save("test_features_pca.npy", test_features_pca)

    # Save labels with error checking
    try:
        np.save("train_labels.npy", train_labels)
        print("train_labels.npy saved successfully")
    except Exception as e:
        print("Failed to save train_labels.npy:", e)
        
    try:
        np.save("test_labels.npy", test_labels)
        print("test_labels.npy saved successfully")
    except Exception as e:
        print("Failed to save test_labels.npy:", e)
    
    print("Data preparation completed.")
