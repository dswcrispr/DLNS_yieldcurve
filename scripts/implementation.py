"""
DLNS (Dynamic Nelson-Siegel) Model Implementation Script

This script implements the complete workflow for training and evaluating
Deep Learning-based Nelson-Siegel yield curve models with various encoder architectures.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import random
import os
import sys
from typing import Dict, List, Tuple, Optional, Any

# Add modules directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from modules.YCdataset import YieldCurveDataset
from modules.encoder import (
    CNN1DTransformerEncoder,
    TransformerOnlyEncoder,
    CNN1DRNNEncoder,
    CNN1DLSTMEncoder
)
from modules.NS_layer import NelsonSiegelLayer
from modules.DLNS import DLNS_CNNTransformer, DLNSTransformerOnly, DLNS_CNNRNN, DLNS_CNNLSTM
from modules.train import train_model, plot_training_history
from modules.evaluation import (
    evaluate_model,
    denormalize_and_evaluate,
    multi_step_forecast,
    evaluate_multi_step_forecast,
    extract_factors_and_lambdas
)
from modules.visualization import (
    plot_yield_curves,
    plot_factors_over_time,
    plot_yield_curve_3d,
    heatmap_predictions_error
)

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)


def set_seed(seed: int = 30) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"All random seeds set to {seed}")


def seed_worker(worker_id: int, seed: int = 30) -> None:
    """
    Worker initialization function for DataLoader to ensure reproducibility.

    Args:
        worker_id: Worker process ID
        seed: Base random seed value
    """
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_and_split_data(
    csv_file: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.07,
    lookback_window: int = 6,
    pred_horizon: int = 6
) -> Tuple[pd.DataFrame, int, int, int]:
    """
    Load yield curve data and calculate train/validation/test split sizes.

    Args:
        csv_file: Path to the processed monthly data CSV file
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        lookback_window: Number of historical time steps to use
        pred_horizon: Number of future time steps to predict

    Returns:
        Tuple of (DataFrame, train_size, val_size, test_size)
    """
    df = pd.read_csv(csv_file, index_col='Date')
    df = df.rename(columns={'3M': '0.25Y', '6M': '0.5Y'})

    train_size = int(train_ratio * len(df))
    val_size = int(val_ratio * len(df))

    print(f"Dataset size: {len(df)}")
    print(f"Train size: {train_size} ({train_ratio*100:.1f}%)")
    print(f"Validation size: {val_size} ({val_ratio*100:.1f}%)")
    print(f"Test size: {len(df) - train_size - val_size} ({(1-train_ratio-val_ratio)*100:.1f}%)")

    return df, train_size, val_size, len(df) - train_size - val_size


def create_datasets(
    df: pd.DataFrame,
    train_size: int,
    val_size: int,
    lookback_window: int,
    pred_horizon: int,
    use_macro: bool
) -> Tuple[YieldCurveDataset, YieldCurveDataset, YieldCurveDataset, Any, Optional[Any]]:
    """
    Create train, validation, and test datasets with optional macro variables.

    Args:
        df: Full DataFrame with yield curves and macro variables
        train_size: Number of samples for training
        val_size: Number of samples for validation
        lookback_window: Number of historical time steps
        pred_horizon: Number of future time steps to predict
        use_macro: Whether to include macroeconomic variables

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, yield_scaler, macro_scaler)
    """
    train_dataset = YieldCurveDataset(
        df.iloc[:train_size],
        seq_length=lookback_window,
        pred_horizon=pred_horizon,
        is_train=True,
        use_macro=use_macro
    )
    yield_scaler = train_dataset.yield_scaler
    macro_scaler = train_dataset.macro_scaler if use_macro else None

    val_dataset = YieldCurveDataset(
        df.iloc[train_size:train_size+val_size],
        seq_length=lookback_window,
        pred_horizon=pred_horizon,
        yield_scaler=yield_scaler,
        macro_scaler=macro_scaler,
        is_train=False,
        use_macro=use_macro
    )

    test_dataset = YieldCurveDataset(
        df.iloc[train_size+val_size:],
        seq_length=lookback_window,
        pred_horizon=pred_horizon,
        yield_scaler=yield_scaler,
        macro_scaler=macro_scaler,
        is_train=False,
        use_macro=use_macro
    )

    return train_dataset, val_dataset, test_dataset, yield_scaler, macro_scaler


def create_data_loaders(
    train_dataset: YieldCurveDataset,
    val_dataset: YieldCurveDataset,
    test_dataset: YieldCurveDataset,
    batch_size: int = 16,
    seed: int = 30
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects for train, validation, and test datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for DataLoader
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    def worker_init(worker_id):
        seed_worker(worker_id, seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(seed)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(seed)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(seed)
    )

    return train_loader, val_loader, test_loader


def initialize_models(
    input_dim: int,
    yield_dim: int,
    macro_dim: int,
    maturities: List[float],
    seq_length: int,
    use_macro: bool,
    hidden_dim: int = 64,
    cnn_out_channels: int = 32,
    cnn_kernel_size: int = 3,
    transformer_dim: int = 64,
    nhead: int = 4,
    num_transformer_layers: int = 2,
    rnn_hidden_dim: int = 64,
    lstm_hidden_dim: int = 64,
    num_rnn_layers: int = 2,
    num_lstm_layers: int = 2,
    dropout: float = 0.1,
    time_varying_decay: bool = True
) -> Dict[str, nn.Module]:
    """
    Initialize all DLNS model variants.

    Args:
        input_dim: Total number of input features
        yield_dim: Number of yield curve maturities
        macro_dim: Number of macroeconomic features
        maturities: List of maturity values in years
        seq_length: Sequence length for time series
        use_macro: Whether to use macroeconomic variables
        hidden_dim: Hidden dimension size
        cnn_out_channels: Number of CNN output channels
        cnn_kernel_size: CNN kernel size
        transformer_dim: Transformer dimension size
        nhead: Number of attention heads
        num_transformer_layers: Number of transformer layers
        rnn_hidden_dim: RNN hidden dimension
        lstm_hidden_dim: LSTM hidden dimension
        num_rnn_layers: Number of RNN layers
        num_lstm_layers: Number of LSTM layers
        dropout: Dropout rate
        time_varying_decay: Whether to use time-varying decay parameter

    Returns:
        Dictionary of initialized models
    """
    models = {}

    models['cnn_transformer'] = DLNS_CNNTransformer(
        input_dim=input_dim,
        yield_dim=yield_dim,
        macro_dim=macro_dim,
        maturities=maturities,
        hidden_dim=hidden_dim,
        time_varying_decay=time_varying_decay,
        cnn_out_channels=cnn_out_channels,
        cnn_kernel_size=cnn_kernel_size,
        transformer_dim=transformer_dim,
        nhead=nhead,
        num_transformer_layers=num_transformer_layers,
        dropout=dropout,
        seq_length=seq_length,
        use_macro=use_macro
    )

    models['transformer_only'] = DLNSTransformerOnly(
        input_dim=input_dim,
        maturities=maturities,
        hidden_dim=hidden_dim,
        time_varying_decay=time_varying_decay,
        transformer_dim=transformer_dim,
        nhead=nhead,
        num_transformer_layers=num_transformer_layers,
        dropout=dropout,
        seq_length=seq_length
    )

    models['cnn_rnn'] = DLNS_CNNRNN(
        input_dim=input_dim,
        yield_dim=yield_dim,
        macro_dim=macro_dim,
        maturities=maturities,
        hidden_dim=hidden_dim,
        time_varying_decay=time_varying_decay,
        cnn_out_channels=cnn_out_channels,
        cnn_kernel_size=cnn_kernel_size,
        rnn_hidden_dim=rnn_hidden_dim,
        num_rnn_layers=num_rnn_layers,
        dropout=dropout,
        seq_length=seq_length,
        use_macro=use_macro,
        bidirectional=False
    )

    models['cnn_lstm'] = DLNS_CNNLSTM(
        input_dim=input_dim,
        yield_dim=yield_dim,
        macro_dim=macro_dim,
        maturities=maturities,
        hidden_dim=hidden_dim,
        time_varying_decay=time_varying_decay,
        cnn_out_channels=cnn_out_channels,
        cnn_kernel_size=cnn_kernel_size,
        lstm_hidden_dim=lstm_hidden_dim,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout,
        seq_length=seq_length,
        use_macro=use_macro
    )

    return models


def train_all_models(
    models: Dict[str, nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    n_epochs: int = 100,
    patience: int = 30,
    save_dir: str = './models'
) -> Dict[str, Dict]:
    """
    Train all model variants and save their training histories.

    Args:
        models: Dictionary of model instances
        train_loader: Training data loader
        val_loader: Validation data loader
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        n_epochs: Maximum number of training epochs
        patience: Early stopping patience
        save_dir: Directory to save model checkpoints

    Returns:
        Dictionary of training histories for each model
    """
    os.makedirs(save_dir, exist_ok=True)
    histories = {}
    criterion = torch.nn.MSELoss()

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name.replace('_', ' ').title()} Model")
        print(f"{'='*60}")

        model_path = os.path.join(save_dir, f'{model_name}_model.pth')

        trained_model, history = train_model(
            model,
            train_loader,
            val_loader,
            criterion=criterion,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            patience=patience,
            model_path=model_path
        )

        models[model_name] = trained_model
        histories[model_name] = history

        plot_training_history(history)
        plt.savefig(f'{save_dir}/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    return histories


def evaluate_all_models(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    yield_scaler: Any,
    maturities: List[float]
) -> Dict[str, Dict]:
    """
    Evaluate all trained models on test data.

    Args:
        models: Dictionary of trained model instances
        test_loader: Test data loader
        yield_scaler: Scaler for denormalizing yield predictions
        maturities: List of maturity values

    Returns:
        Dictionary of evaluation results for each model
    """
    results = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name.replace('_', ' ').title()} Model...")

        result = denormalize_and_evaluate(
            model,
            test_loader,
            yield_scaler,
            maturities
        )

        results[model_name] = result
        print(f"RMSE: {result['rmse']:.6f}")

    return results


def compare_model_performance(
    results: Dict[str, Dict],
    maturities: List[float],
    save_dir: str = './output'
) -> None:
    """
    Compare and visualize performance across all models.

    Args:
        results: Dictionary of evaluation results
        maturities: List of maturity values
        save_dir: Directory to save comparison plots
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Model Performance Comparison (RMSE)")
    print("="*60)

    for model_name, result in results.items():
        print(f"{model_name.replace('_', ' ').title()}: {result['rmse']:.2f}")

    # Plot yield curves for sample predictions
    for model_name, result in results.items():
        plot_yield_curves(
            result['denorm_predictions'],
            result['denorm_targets'],
            maturities,
            indices=[0, 1, 2],
            title=f"{model_name.replace('_', ' ').title()}: Predicted vs Actual Yield Curves"
        )
        plt.savefig(f"{save_dir}/{model_name}_yield_curves.png", dpi=300, bbox_inches='tight')
        plt.close()


def analyze_factors_and_lambdas(
    models: Dict[str, nn.Module],
    train_dataset: YieldCurveDataset,
    val_dataset: YieldCurveDataset,
    test_dataset: YieldCurveDataset,
    df: pd.DataFrame,
    seq_length: int,
    save_dir: str = './output'
) -> None:
    """
    Extract and visualize Nelson-Siegel factors and lambda parameters.

    Args:
        models: Dictionary of trained models
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        df: Original DataFrame for date indexing
        seq_length: Sequence length used in datasets
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    full_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset, test_dataset])
    full_loader = DataLoader(full_dataset, batch_size=16, shuffle=False)

    factors_dict = {}
    lambdas_dict = {}

    for model_name, model in models.items():
        factors, lambdas = extract_factors_and_lambdas(model, full_loader)
        factors_dict[model_name] = factors
        lambdas_dict[model_name] = lambdas

    full_dates = df.index[seq_length:len(full_dataset)+seq_length]

    if len(full_dates) > len(list(factors_dict.values())[0]):
        full_dates = full_dates[:len(list(factors_dict.values())[0])]
    elif len(full_dates) < len(list(factors_dict.values())[0]):
        for key in factors_dict:
            factors_dict[key] = factors_dict[key][:len(full_dates)]
            lambdas_dict[key] = lambdas_dict[key][:len(full_dates)]

    if isinstance(full_dates[0], str):
        full_dates = pd.to_datetime(full_dates)

    fig, axs = plt.subplots(4, 1, figsize=(15, 15), sharex=True)

    factor_names = ['Level', 'Slope', 'Curvature', 'Lambda']
    model_names_display = [name.replace('_', ' ').title() for name in models.keys()]
    colors = ['b', 'r', 'g', 'purple']

    for i, factor_name in enumerate(factor_names):
        ax = axs[i]
        for j, (model_name, display_name) in enumerate(zip(models.keys(), model_names_display)):
            if i < 3:
                data = factors_dict[model_name][:, i]
            else:
                data = lambdas_dict[model_name]

            ax.plot(full_dates, data, color=colors[j], label=display_name, linewidth=1.5)

        ax.set_title(f'{factor_name} Factor Comparison', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    years = mdates.YearLocator(4)
    years_fmt = mdates.DateFormatter("'%y")

    for ax in axs:
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.set_xlim(full_dates[0], full_dates[-1])

    plt.suptitle('Yield Curve Factors Comparison Between Models', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{save_dir}/factor_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def multi_step_evaluation(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    yield_scaler: Any,
    pred_horizon: int,
    yield_dim: int,
    use_macro: bool,
    num_macro: int,
    save_dir: str = './output'
) -> Dict[str, Dict]:
    """
    Evaluate multi-step forecasting performance for all models.

    Args:
        models: Dictionary of trained models
        test_loader: Test data loader
        yield_scaler: Scaler for denormalizing predictions
        pred_horizon: Number of steps to forecast
        yield_dim: Number of yield maturities
        use_macro: Whether macro variables are used
        num_macro: Number of macro variables
        save_dir: Directory to save results

    Returns:
        Dictionary of multi-step evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)
    multi_results = {}

    print("\n" + "="*60)
    print("Multi-Step Forecasting Evaluation")
    print("="*60)

    for model_name, model in models.items():
        print(f"\n{model_name.replace('_', ' ').title()} Multi-step Evaluation:")

        result = evaluate_multi_step_forecast(
            model=model,
            test_loader=test_loader,
            yield_scaler=yield_scaler,
            pred_horizon=pred_horizon,
            yield_dim=yield_dim,
            use_macro=use_macro,
            num_macro=num_macro,
            denormalize=True
        )

        multi_results[model_name] = result

    print("\n" + "="*60)
    print("RMSE by Forecast Horizon (basis points)")
    print("="*60)
    header = f"{'Horizon':<10}"
    for name in models.keys():
        header += f" {name.replace('_', ' ').title():<20}"
    print(header)
    print("-" * len(header))

    for h in range(pred_horizon):
        row = f"{h+1:<10}"
        for model_name in models.keys():
            rmse = multi_results[model_name]['rmse_by_horizon'][h] * 100
            row += f" {rmse:<20.2f}"
        print(row)

    print("\n" + "="*60)
    print("Overall Multi-step RMSE (basis points)")
    print("="*60)
    for model_name in models.keys():
        rmse = multi_results[model_name]['rmse'] * 100
        print(f"{model_name.replace('_', ' ').title()}: {rmse:.2f}")

    return multi_results


def main(use_macro: bool = True, seed: int = 30) -> None:
    """
    Main execution function for the DLNS implementation workflow.

    Args:
        use_macro: Whether to include macroeconomic variables
        seed: Random seed for reproducibility
    """
    set_seed(seed)

    print("="*60)
    print("DLNS Yield Curve Model Implementation")
    print("="*60)

    # Load and split data
    df, train_size, val_size, test_size = load_and_split_data(
        csv_file='df_monthly.csv',
        train_ratio=0.80,
        val_ratio=0.07,
        lookback_window=6,
        pred_horizon=6
    )

    # Create datasets
    lookback_window = 6
    pred_horizon = 6
    macro_vars = 4

    train_dataset, val_dataset, test_dataset, yield_scaler, macro_scaler = create_datasets(
        df=df,
        train_size=train_size,
        val_size=val_size,
        lookback_window=lookback_window,
        pred_horizon=pred_horizon,
        use_macro=use_macro
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=16, seed=seed
    )

    # Extract model parameters
    input_dim = train_dataset[0][0].shape[1]
    yield_dim = len(train_dataset.yield_cols)
    macro_dim = len(train_dataset.macro_cols) if use_macro else 0
    seq_length = train_dataset.seq_length
    maturities = [float(col.replace('Y', '')) for col in train_dataset.yield_cols]

    print(f"\nModel Configuration:")
    print(f"Input dimension: {input_dim}")
    print(f"Yield dimension: {yield_dim}")
    print(f"Macro dimension: {macro_dim}")
    print(f"Sequence length: {seq_length}")
    print(f"Prediction horizon: {pred_horizon}")

    # Initialize models
    models = initialize_models(
        input_dim=input_dim,
        yield_dim=yield_dim,
        macro_dim=macro_dim,
        maturities=maturities,
        seq_length=seq_length,
        use_macro=use_macro
    )

    # Train models
    histories = train_all_models(
        models=models,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.001,
        weight_decay=1e-5,
        n_epochs=100,
        patience=30
    )

    # Evaluate models
    results = evaluate_all_models(
        models=models,
        test_loader=test_loader,
        yield_scaler=yield_scaler,
        maturities=maturities
    )

    # Compare performance
    compare_model_performance(results, maturities)

    # Analyze factors and lambdas
    analyze_factors_and_lambdas(
        models=models,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        df=df,
        seq_length=seq_length
    )

    # Multi-step evaluation
    multi_results = multi_step_evaluation(
        models=models,
        test_loader=test_loader,
        yield_scaler=yield_scaler,
        pred_horizon=pred_horizon,
        yield_dim=yield_dim,
        use_macro=use_macro,
        num_macro=macro_vars
    )

    print("\n" + "="*60)
    print("DLNS Implementation Complete!")
    print("="*60)


if __name__ == '__main__':
    # Run with macro variables
    print("Running with macroeconomic variables...")
    main(use_macro=True, seed=30)

    # Uncomment to run without macro variables
    # print("\nRunning without macroeconomic variables...")
    # main(use_macro=False, seed=30)
