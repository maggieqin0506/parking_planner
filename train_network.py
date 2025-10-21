def train_model(data_file='data/training_data.pkl', 
                model_save_path='models/neural_heuristic.pth'):
    """Train the neural network"""
    
    # Load data
    print("Loading training data...")
    with open(data_file, 'rb') as f:
        dataset = pickle.load(f)
    
    features = np.array(dataset['features'])
    labels = np.array(dataset['labels'])
    
    print(f"Loaded {len(features)} training samples")
    
    if len(features) == 0:
        print("ERROR: No training data available!")
        print("Please run data_generator.py first")
        return None
    
    # Handle single sample case
    if len(features) == 1:
        print("WARNING: Only 1 sample, creating minimal model")
        model = NeuralHeuristic(input_size=11, hidden_size=32)
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        return model
    
    # Split train/validation
    n_train = max(1, int(0.8 * len(features)))
    indices = np.random.permutation(len(features))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    # Handle case where val set would be empty
    if len(val_idx) == 0:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    
    train_dataset = ParkingDataset(features[train_idx], labels[train_idx])
    val_dataset = ParkingDataset(features[val_idx], labels[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=min(Config.nn_batch_size, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(Config.nn_batch_size, len(val_dataset)))
    
    # Initialize model
    model = NeuralHeuristic(input_size=11, hidden_size=Config.nn_hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.nn_learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("Training neural network...")
    for epoch in range(Config.nn_epochs):
        # Training
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
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                predictions = model(features_batch)
                loss = criterion(predictions, labels_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{Config.nn_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
    
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {model_save_path}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Neural Network Training History')
    plt.legend()
    plt.grid(True)
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_history.png')
    plt.close()
    
    return model