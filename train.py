import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
from tqdm.notebook import tqdm

# Function to train the model with early stopping based on validation loss
# Includes learning rate scheduling and model checkpointing
# Returns the trained model and a history of training/validation loss

def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion=nn.MSELoss(), 
    learning_rate=0.001,
    weight_decay=1e-5,
    n_epochs=100, 
    patience=20,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    scheduler_factor=0.5,  
    scheduler_patience=10,
    model_path='best_model.pth',
    verbose=True
):
    """
    Train the model with early stopping based on validation loss.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        n_epochs: Maximum number of epochs to train
        patience: Number of epochs to wait for improvement before early stopping
        device: Device to train on (cuda/cpu)
        scheduler_factor: Factor by which to reduce learning rate on plateau
        scheduler_patience: Epochs to wait before reducing learning rate
        model_path: Where to save the best model
        verbose: Whether to print progress
        
    Returns:
        model: The trained model
        history: Dictionary containing training history
    """
    # Move model to the specified device (GPU or CPU)
    model = model.to(device)
    
    # Initialize the optimizer with model parameters, learning rate, and weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler to reduce learning rate when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=scheduler_factor, 
        patience=scheduler_patience, 
        verbose=verbose
    )
    
    # Initialize history dictionary to store training and validation loss
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    best_val_loss = float('inf')  # Track the best validation loss
    counter = 0  # Counter for early stopping
    
    # Training loop over epochs
    for epoch in range(n_epochs):
        # Set model to training mode
        model.train() # dropout and batchnorm are turned on
        train_losses = []
        
        # Use tqdm for progress bar if verbose is True
        train_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Train]') if verbose else train_loader
        
        # Iterate over batches in the training data
        for X_batch, y_batch in train_iterator: # X_batch: (batch_size, 12, input_dim), y_batch: (batch_size, 6, num_yields)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Zero the gradients, this is done to avoid accumulation of gradients
            optimizer.zero_grad()
            
            # Forward pass: compute predicted outputs by passing inputs to the model
            y_pred, factors, lambda_value = model(X_batch)
            
            # Compute the loss
            # Here, we use the first horizon predictions (1-month ahead) for simplicity
            # Check point: loss is computed on the first horizon predictions?
            loss = criterion(y_pred, y_batch[:, 0])
            
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Perform a single optimization step (parameter update)
            optimizer.step()
            
            # Record the training loss
            train_losses.append(loss.item())
        
        # Calculate average training loss for the epoch
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_losses = []
        
        # Use tqdm for progress bar if verbose is True
        val_iterator = tqdm(val_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Valid]') if verbose else val_loader
        
        # Disable gradient computation for validation
        with torch.no_grad():
            for X_batch, y_batch in val_iterator:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                y_pred, factors, lambda_value = model(X_batch)
                
                # Compute the validation loss
                loss = criterion(y_pred, y_batch[:, 0])
                val_losses.append(loss.item())
        
        # Calculate average validation loss for the epoch
        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        
        # Update learning rate scheduler based on validation loss
        scheduler.step(avg_val_loss)
        
        # Print progress if verbose is True
        if verbose:
            print(f'Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Check for improvement in validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0  # Reset counter if validation loss improves
            
            # Save the best model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, model_path)
            
            if verbose:
                print(f'Model improved, saved checkpoint to {model_path}')
        else:
            counter += 1  # Increment counter if no improvement
            if verbose:
                print(f'No improvement for {counter} epochs')
        
        # Early stopping if no improvement for 'patience' epochs
        if counter >= patience:
            if verbose:
                print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load the best model from checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history

# Function to plot the training and validation loss history
# Displays the loss over epochs and the learning rate schedule

def plot_training_history(history):
    """
    Plot the training and validation loss history
    
    Args:
        history: Dictionary containing training history
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot learning rate
    plt.subplot(1, 2, 2)
    plt.plot(history['lr'])
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.show();