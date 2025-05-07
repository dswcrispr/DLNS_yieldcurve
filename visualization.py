import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


def plot_yield_curves(predictions, targets, maturities, indices=None, title='Yield Curve Comparison'):
    """
    Plot predicted vs actual yield curves for specific indices.
    
    Args:
        predictions: Array of predicted yield curves
        targets: Array of target yield curves
        maturities: List of maturities
        indices: List of indices to plot (defaults to first 3)
        title: Plot title
    """
    if indices is None:
        indices = list(range(min(3, len(predictions))))
    
    plt.figure(figsize=(15, 5 * len(indices)))
    
    for i, idx in enumerate(indices):
        plt.subplot(len(indices), 1, i + 1)
        plt.plot(maturities, predictions[idx], 'r-', label='Predicted')
        plt.plot(maturities, targets[idx], 'b-', label='Actual')
        plt.xlabel('Maturity')
        plt.ylabel('Yield')
        plt.title(f'{title} - Sample {idx}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_factors_over_time(factors, lambdas, dates):
    """
    Plot the predicted Nelson-Siegel factors over time.
    
    Args:
        factors: Array of factors [n_samples, n_factors]
        lambdas: Array of lambda values
        dates: List of dates corresponding to the predictions
    """
    factor_names = ['Level', 'Slope', 'Curvature']
    
    plt.figure(figsize=(15, 10))
    
    # Plot first three factors
    for i in range(3):
        plt.subplot(4, 1, i + 1)
        plt.plot(dates, factors[:, i])
        plt.ylabel(factor_names[i])
        plt.grid(True)
        if i == 0:
            plt.title('Nelson-Siegel Factors Over Time')
    
    # Plot lambda
    plt.subplot(4, 1, 4)
    plt.plot(dates, lambdas)
    plt.ylabel('Lambda')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()




def plot_yield_curve_3d(predictions, maturities, dates):
    """
    Create a 3D plot of yield curves over time.
    
    Args:
        predictions: Array of predicted yield curves [n_dates, n_maturities]
        maturities: List of maturities
        dates: List of dates corresponding to the predictions
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(maturities, range(len(dates)))
    
    # Plot surface
    surf = ax.plot_surface(X, Y, predictions, cmap='viridis', edgecolor='none')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Set labels
    ax.set_xlabel('Maturity')
    ax.set_ylabel('Time')
    ax.set_zlabel('Yield')
    
    # Format time axis with date labels
    time_ticks = np.linspace(0, len(dates)-1, min(10, len(dates)))
    time_ticklabels = [dates[int(i)] for i in time_ticks]
    ax.set_yticks(time_ticks)
    ax.set_yticklabels(time_ticklabels)
    
    plt.title('Yield Curve Evolution Over Time')
    plt.tight_layout()
    plt.show()

def plot_factor_loadings(maturities, lambda_value):
    """
    Plot the Nelson-Siegel factor loadings for a given lambda value.
    
    Args:
        maturities: List of maturities
        lambda_value: Lambda decay parameter
    """
    # Calculate factor loadings
    level_loading = np.ones_like(maturities)
    exp_term = np.exp(-lambda_value * np.array(maturities))
    common_term = (1 - exp_term) / (lambda_value * np.array(maturities))
    slope_loading = common_term
    curvature_loading = common_term - exp_term
    
    plt.figure(figsize=(12, 6))
    plt.plot(maturities, level_loading, label='Level', linewidth=2)
    plt.plot(maturities, slope_loading, label='Slope', linewidth=2)
    plt.plot(maturities, curvature_loading, label='Curvature', linewidth=2)
    
    plt.xlabel('Maturity')
    plt.ylabel('Loading')
    plt.title(f'Nelson-Siegel Factor Loadings (λ = {lambda_value:.4f})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def heatmap_predictions_error(predictions, targets, maturities, dates, cmap='RdBu_r'):
    """
    Create a heatmap of prediction errors (residuals) across maturities and time.
    
    Args:
        predictions: Array of predicted yield curves [n_dates, n_maturities]
        targets: Array of actual yield curves [n_dates, n_maturities]
        maturities: List of maturities
        dates: List of dates corresponding to the predictions
        cmap: Colormap to use
    """
    # Calculate errors
    errors = targets - predictions
    
    plt.figure(figsize=(15, 8))
    plt.imshow(errors, aspect='auto', cmap=cmap, interpolation='none')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Prediction Error (actual - predicted)')
    
    # Set axis labels and ticks
    plt.xlabel('Maturity')
    plt.ylabel('Date')
    
    # Format x-axis with maturity labels
    plt.xticks(range(len(maturities)), [str(m) for m in maturities], rotation=90)
    
    # Format y-axis with date labels
    n_date_ticks = min(20, len(dates))
    date_indices = np.linspace(0, len(dates)-1, n_date_ticks, dtype=int)
    plt.yticks(date_indices, [dates[i] for i in date_indices])
    
    plt.title('Yield Curve Prediction Errors')
    plt.tight_layout()
    plt.show()