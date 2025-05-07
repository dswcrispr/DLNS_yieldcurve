import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_factors = []
    all_lambdas = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            
            # Forward pass
            y_pred, factors, lambda_value = model(X_batch)
            
            # Store predictions and targets
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch[:, 0].cpu().numpy())  # First horizon (1-month ahead)
            all_factors.append(factors.cpu().numpy())
            all_lambdas.append(lambda_value.cpu().numpy())
    
    # Concatenate batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_factors = np.vstack(all_factors)
    all_lambdas = np.concatenate(all_lambdas)
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    
    # Calculate metrics by maturity
    mse_by_maturity = np.mean((all_targets - all_preds) ** 2, axis=0)
    rmse_by_maturity = np.sqrt(mse_by_maturity)
    mae_by_maturity = np.mean(np.abs(all_targets - all_preds), axis=0)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mse_by_maturity': mse_by_maturity,
        'rmse_by_maturity': rmse_by_maturity,
        'mae_by_maturity': mae_by_maturity,
        'predictions': all_preds,
        'targets': all_targets,
        'factors': all_factors,
        'lambdas': all_lambdas
    }

def denormalize_and_evaluate(model, test_loader, yield_scaler, maturities,
                            device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate the model on test data with denormalization of predictions.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        yield_scaler: Scaler used to normalize yield data
        maturities: List of maturities used in the model
        device: Device to evaluate on
        
    Returns:
        dict: Dictionary containing evaluation metrics on denormalized data
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_factors = []
    all_lambdas = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            
            # Forward pass
            y_pred, factors, lambda_value = model(X_batch)
            
            # Store normalized predictions and targets
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch[:, 0].cpu().numpy())  # First horizon (1-month ahead)
            all_factors.append(factors.cpu().numpy())
            all_lambdas.append(lambda_value.cpu().numpy())
    
    # Concatenate batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_factors = np.vstack(all_factors)
    all_lambdas = np.concatenate(all_lambdas)
    
    # Denormalize predictions and targets
    denorm_preds = yield_scaler.inverse_transform(all_preds.reshape(-1, 1)).reshape(all_preds.shape)
    denorm_targets = yield_scaler.inverse_transform(all_targets.reshape(-1, 1)).reshape(all_targets.shape)
    
    # Calculate metrics on denormalized data
    mse = mean_squared_error(denorm_targets, denorm_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(denorm_targets, denorm_preds)
    
    # Calculate metrics by maturity
    mse_by_maturity = np.mean((denorm_targets - denorm_preds) ** 2, axis=0)
    rmse_by_maturity = np.sqrt(mse_by_maturity)
    mae_by_maturity = np.mean(np.abs(denorm_targets - denorm_preds), axis=0)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mse_by_maturity': mse_by_maturity,
        'rmse_by_maturity': rmse_by_maturity,
        'mae_by_maturity': mae_by_maturity,
        'denorm_predictions': denorm_preds,
        'denorm_targets': denorm_targets,
        'factors': all_factors,
        'lambdas': all_lambdas
    }


def multi_step_forecast(model, input_data, pred_horizon=6, yield_dim=None, use_macro=True,
                        num_macro=3,device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generate multi-step forecasts using the trained model.
    
    Args:
        model: Trained model
        input_data: Input tensor of shape [batch_size, seq_length, features]
        pred_horizon: Number of steps ahead to forecast
        yield_dim: Number of yield dimensions (maturities), If None, will be inferred based on use_macro
        use_macro: Whether macro variables are used in the model
        device: Device to evaluate on
        
    Returns:
        np.array: Array of predictions for each forecast horizon
    """
    model = model.to(device)
    model.eval()
    input_data = input_data.to(device)

    # Infer yield dimension if not provided
    if yield_dim is None:
        if use_macro:
            # If using macro variables, assume the last 3 columns are macro variables
            yield_dim = input_data.shape[2] - num_macro
        else:
            # If not using macro variables, all features are yields
            yield_dim = input_data.shape[2]
    
    with torch.no_grad():
        # Get initial forecast
        forecasts = []
        current_input = input_data.clone() # clone() creates a copy of the input tensor
        
        for step in range(pred_horizon):
            # Forward pass
            y_pred, factors, lambda_value = model(current_input)
            forecasts.append(y_pred.cpu().numpy())
            
            # Update input for next step (roll forward)
            # Assuming batch_size=1 for simplicity in this example
            if step < pred_horizon - 1:
                # Remove oldest time step
                new_input = current_input[:, 1:, :]
                
                # Add forecast as newest time step
                # This assumes y_pred contains yields at same positions as in the input
                # Create new last row with predicted yields and previous macro variables
                new_last_row = torch.zeros_like(current_input[:, 0, :])
                new_last_row[:, :yield_dim] = y_pred  # Predicted yields

                if use_macro:
                    # Keep macro vars only if we are using them
                    new_last_row[:, yield_dim:] = current_input[:, -1, yield_dim:]  
                
                # Concatenate to form new input
                new_last_row = new_last_row.unsqueeze(1)
                current_input = torch.cat([new_input, new_last_row], dim=1)
    
    return np.array(forecasts)