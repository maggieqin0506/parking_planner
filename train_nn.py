#!/usr/bin/env python3
"""
Train neural network - WORKING VERSION
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys

def main():
    print("=" * 60, flush=True)
    print("NEURAL NETWORK TRAINING", flush=True)
    print("=" * 60, flush=True)
    
    # Load data
    print("\n1. Loading training data...", flush=True)
    try:
        with open('data/training_data.pkl', 'rb') as f:
            dataset = pickle.load(f)
        features = np.array(dataset['features'])
        labels = np.array(dataset['labels'])
        print(f"   ✓ Loaded {len(features)} samples", flush=True)
    except Exception as e:
        print(f"   ✗ ERROR loading data: {e}", flush=True)
        sys.exit(1)
    
    # Dataset class
    class ParkingDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.FloatTensor(features)
            self.labels = torch.FloatTensor(labels).unsqueeze(1)
            self.features[:, 0:2] /= 15.0
            self.features[:, 4] /= 20.0
            self.features[:, 5:9] /= 5.0
            self.features[:, 9] /= 2.0
        
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
    # Model
    class NeuralHeuristic(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(11, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Prepare data
    print("\n2. Preparing datasets...", flush=True)
    n_train = int(0.8 * len(features))
    indices = np.random.permutation(len(features))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    train_dataset = ParkingDataset(features[train_idx], labels[train_idx])
    val_dataset = ParkingDataset(features[val_idx], labels[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    print(f"   ✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)
    
    # Setup
    print("\n3. Initializing model...", flush=True)
    model = NeuralHeuristic()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("   ✓ Model ready", flush=True)
    
    # Training
    print("\n4. Training (50 epochs)...", flush=True)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(50):
        model.train()
        train_loss = 0
        for features_batch, labels_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(features_batch)
            loss = criterion(predictions, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                predictions = model(features_batch)
                loss = criterion(predictions, labels_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/50 - Train: {train_loss:.4f}, Val: {val_loss:.4f}", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/neural_heuristic.pth')
    
    print(f"\n   ✓ Best validation loss: {best_val_loss:.4f}", flush=True)
    
    # Save plot
    print("\n5. Saving training history...", flush=True)
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Neural Network Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_history.png')
    plt.close()
    print("   ✓ Saved: results/training_history.png", flush=True)
    
    print("\n" + "=" * 60, flush=True)
    print("✓ TRAINING COMPLETE!", flush=True)
    print("=" * 60, flush=True)
    print("\nFiles created:", flush=True)
    print(f"  - models/neural_heuristic.pth", flush=True)
    print("  - results/training_history.png", flush=True)
    print("\nNext step: python main.py", flush=True)

if __name__ == '__main__':
    main()