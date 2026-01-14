"""
DLNS Yield Curve Modeling - Main Pipeline Script

This script orchestrates the complete end-to-end pipeline for training and evaluating
Dynamic Nelson-Siegel yield curve models with deep learning architectures.

Usage:
    python main.py [--config CONFIG_FILE] [--use-macro] [--no-macro] [--seed SEED]

Author: DLNS Project
Date: 2025
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures

# Add modules directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from modules.data_preprocessing import process_yield_curve_data
from modules.YCdataset import YieldCurveDataset
from modules.DLNS import DLNS_CNNTransformer, DLNSTransformerOnly, DLNS_CNNRNN, DLNS_CNNLSTM
from modules.train import train_model, plot_training_history
from modules.evaluation import (
    denormalize_and_evaluate,
    multi_step_forecast,
    evaluate_multi_step_forecast,
    extract_factors_and_lambdas
)
from modules.visualization import plot_yield_curves, heatmap_predictions_error

from torch.utils.data import DataLoader
import random
import warnings
warnings.filterwarnings('ignore')


class DLNSPipeline:
    """
    Main pipeline class for DLNS yield curve modeling.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DLNS pipeline.

        Args:
            config: Configuration dictionary with all parameters
        """
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup directories
        self.setup_directories()

        # Setup logging
        self.setup_logging()

        # Log configuration
        self.logger.info("="*80)
        self.logger.info("DLNS Yield Curve Modeling Pipeline")
        self.logger.info("="*80)
        self.logger.info(f"Timestamp: {self.timestamp}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")

        # Set random seeds
        self.set_seed(config['seed'])

        # Initialize storage for results
        self.results = {}
        self.models = {}
        self.histories = {}

    def setup_directories(self):
        """Create necessary output directories."""
        self.output_dir = Path(self.config['output_dir'])
        self.run_dir = self.output_dir / f"run_{self.timestamp}"

        # Create subdirectories
        self.dirs = {
            'root': self.run_dir,
            'models': self.run_dir / 'models',
            'plots': self.run_dir / 'plots',
            'data': self.run_dir / 'data',
            'logs': self.run_dir / 'logs',
            'results': self.run_dir / 'results'
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Configure logging to file and console."""
        log_file = self.dirs['logs'] / 'pipeline.log'

        # Create logger
        self.logger = logging.getLogger('DLNS')
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        self.logger.info(f"Random seed set to {seed}")

    def seed_worker(self, worker_id: int):
        """Worker initialization function for DataLoader."""
        worker_seed = self.config['seed'] + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def step1_load_and_process_data(self):
        """Step 1: Load and process yield curve data."""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 1: Data Loading and Processing")
        self.logger.info("="*80)

        try:
            # Check if processed data already exists
            processed_file = self.config['processed_data_file']

            if os.path.exists(processed_file):
                self.logger.info(f"Loading existing processed data from {processed_file}")
                self.df = pd.read_csv(processed_file, index_col='Date')
            else:
                self.logger.info(f"Processing raw data from {self.config['raw_data_file']}")
                self.df = process_yield_curve_data(self.config['raw_data_file'])
                self.logger.info(f"Processed data saved to {processed_file}")

            # Rename columns
            self.df = self.df.rename(columns={'3M': '0.25Y', '6M': '0.5Y'})

            # Save processed data to run directory
            processed_copy = self.dirs['data'] / 'df_monthly.csv'
            self.df.to_csv(processed_copy)

            self.logger.info(f"Data shape: {self.df.shape}")
            self.logger.info(f"Date range: {self.df.index[0]} to {self.df.index[-1]}")
            self.logger.info(f"Columns: {list(self.df.columns)}")

            return True

        except Exception as e:
            self.logger.error(f"Error in data processing: {str(e)}", exc_info=True)
            return False

    def step2_create_datasets(self):
        """Step 2: Create train/validation/test datasets."""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 2: Dataset Creation and Splitting")
        self.logger.info("="*80)

        try:
            # Calculate split sizes
            train_size = int(self.config['train_ratio'] * len(self.df))
            val_size = int(self.config['val_ratio'] * len(self.df))
            test_size = len(self.df) - train_size - val_size

            self.logger.info(f"Total samples: {len(self.df)}")
            self.logger.info(f"Train: {train_size} ({self.config['train_ratio']*100:.1f}%)")
            self.logger.info(f"Validation: {val_size} ({self.config['val_ratio']*100:.1f}%)")
            self.logger.info(f"Test: {test_size} ({(1-self.config['train_ratio']-self.config['val_ratio'])*100:.1f}%)")

            # Create datasets
            self.train_dataset = YieldCurveDataset(
                self.df.iloc[:train_size],
                seq_length=self.config['lookback_window'],
                pred_horizon=self.config['pred_horizon'],
                is_train=True,
                use_macro=self.config['use_macro']
            )

            self.yield_scaler = self.train_dataset.yield_scaler
            self.macro_scaler = self.train_dataset.macro_scaler if self.config['use_macro'] else None

            self.val_dataset = YieldCurveDataset(
                self.df.iloc[train_size:train_size+val_size],
                seq_length=self.config['lookback_window'],
                pred_horizon=self.config['pred_horizon'],
                yield_scaler=self.yield_scaler,
                macro_scaler=self.macro_scaler,
                is_train=False,
                use_macro=self.config['use_macro']
            )

            self.test_dataset = YieldCurveDataset(
                self.df.iloc[train_size+val_size:],
                seq_length=self.config['lookback_window'],
                pred_horizon=self.config['pred_horizon'],
                yield_scaler=self.yield_scaler,
                macro_scaler=self.macro_scaler,
                is_train=False,
                use_macro=self.config['use_macro']
            )

            # Create data loaders
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                worker_init_fn=self.seed_worker,
                generator=torch.Generator().manual_seed(self.config['seed'])
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config['batch_size'],
                worker_init_fn=self.seed_worker,
                generator=torch.Generator().manual_seed(self.config['seed'])
            )

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config['batch_size'],
                worker_init_fn=self.seed_worker,
                generator=torch.Generator().manual_seed(self.config['seed'])
            )

            # Extract model parameters
            self.input_dim = self.train_dataset[0][0].shape[1]
            self.yield_dim = len(self.train_dataset.yield_cols)
            self.macro_dim = len(self.train_dataset.macro_cols) if self.config['use_macro'] else 0
            self.seq_length = self.train_dataset.seq_length
            self.maturities = [float(col.replace('Y', '')) for col in self.train_dataset.yield_cols]

            self.logger.info(f"Input dimension: {self.input_dim}")
            self.logger.info(f"Yield dimension: {self.yield_dim}")
            self.logger.info(f"Macro dimension: {self.macro_dim}")
            self.logger.info(f"Sequence length: {self.seq_length}")
            self.logger.info(f"Maturities: {self.maturities}")

            return True

        except Exception as e:
            self.logger.error(f"Error in dataset creation: {str(e)}", exc_info=True)
            return False

    def step3_initialize_models(self):
        """Step 3: Initialize all model architectures."""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 3: Model Initialization")
        self.logger.info("="*80)

        try:
            model_configs = self.config['model_params']

            # Initialize CNN-Transformer model
            self.logger.info("Initializing CNN-Transformer model...")
            self.models['cnn_transformer'] = DLNS_CNNTransformer(
                input_dim=self.input_dim,
                yield_dim=self.yield_dim,
                macro_dim=self.macro_dim,
                maturities=self.maturities,
                hidden_dim=model_configs['hidden_dim'],
                time_varying_decay=model_configs['time_varying_decay'],
                cnn_out_channels=model_configs['cnn_out_channels'],
                cnn_kernel_size=model_configs['cnn_kernel_size'],
                transformer_dim=model_configs['transformer_dim'],
                nhead=model_configs['nhead'],
                num_transformer_layers=model_configs['num_transformer_layers'],
                dropout=model_configs['dropout'],
                seq_length=self.seq_length,
                use_macro=self.config['use_macro']
            )

            # Initialize Transformer-Only model
            self.logger.info("Initializing Transformer-Only model...")
            self.models['transformer_only'] = DLNSTransformerOnly(
                input_dim=self.input_dim,
                maturities=self.maturities,
                hidden_dim=model_configs['hidden_dim'],
                time_varying_decay=model_configs['time_varying_decay'],
                transformer_dim=model_configs['transformer_dim'],
                nhead=model_configs['nhead'],
                num_transformer_layers=model_configs['num_transformer_layers'],
                dropout=model_configs['dropout'],
                seq_length=self.seq_length
            )

            # Initialize CNN-RNN model
            self.logger.info("Initializing CNN-RNN model...")
            self.models['cnn_rnn'] = DLNS_CNNRNN(
                input_dim=self.input_dim,
                yield_dim=self.yield_dim,
                macro_dim=self.macro_dim,
                maturities=self.maturities,
                hidden_dim=model_configs['hidden_dim'],
                time_varying_decay=model_configs['time_varying_decay'],
                cnn_out_channels=model_configs['cnn_out_channels'],
                cnn_kernel_size=model_configs['cnn_kernel_size'],
                rnn_hidden_dim=model_configs['rnn_hidden_dim'],
                num_rnn_layers=model_configs['num_rnn_layers'],
                dropout=model_configs['dropout'],
                seq_length=self.seq_length,
                use_macro=self.config['use_macro'],
                bidirectional=False
            )

            # Initialize CNN-LSTM model
            self.logger.info("Initializing CNN-LSTM model...")
            self.models['cnn_lstm'] = DLNS_CNNLSTM(
                input_dim=self.input_dim,
                yield_dim=self.yield_dim,
                macro_dim=self.macro_dim,
                maturities=self.maturities,
                hidden_dim=model_configs['hidden_dim'],
                time_varying_decay=model_configs['time_varying_decay'],
                cnn_out_channels=model_configs['cnn_out_channels'],
                cnn_kernel_size=model_configs['cnn_kernel_size'],
                lstm_hidden_dim=model_configs['lstm_hidden_dim'],
                num_lstm_layers=model_configs['num_lstm_layers'],
                dropout=model_configs['dropout'],
                seq_length=self.seq_length,
                use_macro=self.config['use_macro']
            )

            # Log model parameter counts
            for name, model in self.models.items():
                param_count = sum(p.numel() for p in model.parameters())
                trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.logger.info(f"{name}: {param_count:,} parameters ({trainable_count:,} trainable)")

            return True

        except Exception as e:
            self.logger.error(f"Error in model initialization: {str(e)}", exc_info=True)
            return False

    def step4_train_models(self):
        """Step 4: Train all models."""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 4: Model Training")
        self.logger.info("="*80)

        try:
            training_config = self.config['training']
            criterion = torch.nn.MSELoss()

            for model_name, model in self.models.items():
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Training {model_name.replace('_', ' ').title()} Model")
                self.logger.info(f"{'='*60}")

                model_path = self.dirs['models'] / f'{model_name}_model.pth'

                trained_model, history = train_model(
                    model,
                    self.train_loader,
                    self.val_loader,
                    criterion=criterion,
                    learning_rate=training_config['learning_rate'],
                    weight_decay=training_config['weight_decay'],
                    n_epochs=training_config['n_epochs'],
                    patience=training_config['patience'],
                    model_path=str(model_path),
                    verbose=True
                )

                self.models[model_name] = trained_model
                self.histories[model_name] = history

                # Plot and save training history
                fig = plot_training_history(history)
                plot_path = self.dirs['plots'] / f'{model_name}_training_history.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Training history saved to {plot_path}")

                # Save history to JSON
                history_file = self.dirs['results'] / f'{model_name}_history.json'
                with open(history_file, 'w') as f:
                    json.dump(history, f, indent=2)

            return True

        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}", exc_info=True)
            return False

    def step5_evaluate_models(self):
        """Step 5: Evaluate models on test set."""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 5: Model Evaluation")
        self.logger.info("="*80)

        try:
            for model_name, model in self.models.items():
                self.logger.info(f"\nEvaluating {model_name.replace('_', ' ').title()} Model...")

                result = denormalize_and_evaluate(
                    model,
                    self.test_loader,
                    self.yield_scaler,
                    self.maturities
                )

                self.results[model_name] = result

                # Log metrics
                self.logger.info(f"  RMSE: {result['rmse']:.6f}")
                self.logger.info(f"  MAE: {result['mae']:.6f}")
                self.logger.info(f"  MSE: {result['mse']:.6f}")

                # Plot yield curves
                plot_yield_curves(
                    result['denorm_predictions'],
                    result['denorm_targets'],
                    self.maturities,
                    indices=[0, 1, 2],
                    title=f"{model_name.replace('_', ' ').title()}: Predicted vs Actual"
                )
                plot_path = self.dirs['plots'] / f'{model_name}_yield_curves.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

                # Plot error heatmap
                test_dates = self.df.iloc[-len(result['denorm_predictions']):].index
                heatmap_predictions_error(
                    result['denorm_predictions'],
                    result['denorm_targets'],
                    self.maturities,
                    test_dates
                )
                heatmap_path = self.dirs['plots'] / f'{model_name}_error_heatmap.png'
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()

            # Create comparison summary
            self.create_comparison_summary()

            return True

        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}", exc_info=True)
            return False

    def step6_multi_step_forecast(self):
        """Step 6: Multi-step forecasting evaluation."""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 6: Multi-Step Forecasting")
        self.logger.info("="*80)

        try:
            multi_results = {}

            for model_name, model in self.models.items():
                self.logger.info(f"\n{model_name.replace('_', ' ').title()} Multi-step Evaluation:")

                result = evaluate_multi_step_forecast(
                    model=model,
                    test_loader=self.test_loader,
                    yield_scaler=self.yield_scaler,
                    pred_horizon=self.config['pred_horizon'],
                    yield_dim=self.yield_dim,
                    use_macro=self.config['use_macro'],
                    num_macro=self.macro_dim,
                    denormalize=True
                )

                multi_results[model_name] = result

                # Log RMSE by horizon
                self.logger.info("  RMSE by forecast horizon (basis points):")
                for h in range(self.config['pred_horizon']):
                    rmse = result['rmse_by_horizon'][h] * 100
                    self.logger.info(f"    Horizon {h+1}: {rmse:.2f}")

                self.logger.info(f"  Overall RMSE: {result['rmse'] * 100:.2f} basis points")

            # Save multi-step results
            self.save_multi_step_results(multi_results)

            # Create comparison plots
            self.plot_multi_step_comparison(multi_results)

            return True

        except Exception as e:
            self.logger.error(f"Error in multi-step forecasting: {str(e)}", exc_info=True)
            return False

    def create_comparison_summary(self):
        """Create and save model comparison summary."""
        self.logger.info("\n" + "="*80)
        self.logger.info("Model Performance Comparison")
        self.logger.info("="*80)

        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'MSE': result['mse']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('RMSE')

        # Log comparison
        self.logger.info("\n" + comparison_df.to_string(index=False))

        # Save to CSV
        comparison_file = self.dirs['results'] / 'model_comparison.csv'
        comparison_df.to_csv(comparison_file, index=False)
        self.logger.info(f"\nComparison saved to {comparison_file}")

        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df.plot(x='Model', y=['RMSE', 'MAE'], kind='bar', ax=ax)
        ax.set_ylabel('Error')
        ax.set_title('Model Performance Comparison')
        ax.legend(['RMSE', 'MAE'])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot_path = self.dirs['plots'] / 'model_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_multi_step_results(self, multi_results: Dict):
        """Save multi-step forecasting results."""
        # Create summary DataFrame
        summary_data = []
        for model_name, result in multi_results.items():
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Overall_RMSE_bps': result['rmse'] * 100,
                **{f'Horizon_{h+1}_RMSE_bps': result['rmse_by_horizon'][h] * 100
                   for h in range(len(result['rmse_by_horizon']))}
            })

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.dirs['results'] / 'multi_step_forecast_results.csv'
        summary_df.to_csv(summary_file, index=False)
        self.logger.info(f"Multi-step results saved to {summary_file}")

    def plot_multi_step_comparison(self, multi_results: Dict):
        """Create multi-step forecasting comparison plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot RMSE by horizon
        for model_name, result in multi_results.items():
            horizons = range(1, len(result['rmse_by_horizon']) + 1)
            rmse_bps = [rmse * 100 for rmse in result['rmse_by_horizon']]
            ax1.plot(horizons, rmse_bps, marker='o', label=model_name.replace('_', ' ').title())

        ax1.set_xlabel('Forecast Horizon (months)')
        ax1.set_ylabel('RMSE (basis points)')
        ax1.set_title('RMSE by Forecast Horizon')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot RMSE by maturity
        for model_name, result in multi_results.items():
            rmse_by_mat = result['rmse_by_maturity'] * 100
            ax2.plot(range(len(rmse_by_mat)), rmse_by_mat, marker='o',
                    label=model_name.replace('_', ' ').title())

        ax2.set_xlabel('Maturity Index')
        ax2.set_ylabel('RMSE (basis points)')
        ax2.set_title('RMSE by Maturity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.dirs['plots'] / 'multi_step_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Multi-step comparison plot saved to {plot_path}")

    def save_final_summary(self):
        """Save final pipeline summary."""
        self.logger.info("\n" + "="*80)
        self.logger.info("Saving Final Summary")
        self.logger.info("="*80)

        summary = {
            'timestamp': self.timestamp,
            'config': self.config,
            'data_shape': self.df.shape,
            'date_range': {
                'start': str(self.df.index[0]),
                'end': str(self.df.index[-1])
            },
            'models': list(self.models.keys()),
            'results': {
                name: {
                    'rmse': float(result['rmse']),
                    'mae': float(result['mae']),
                    'mse': float(result['mse'])
                }
                for name, result in self.results.items()
            },
            'output_directory': str(self.run_dir)
        }

        summary_file = self.dirs['results'] / 'pipeline_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Pipeline summary saved to {summary_file}")
        self.logger.info(f"All outputs saved to {self.run_dir}")

    def run(self):
        """Execute the complete pipeline."""
        start_time = datetime.now()
        self.logger.info(f"Pipeline started at {start_time}")

        try:
            # Execute pipeline steps
            steps = [
                ('Data Loading', self.step1_load_and_process_data),
                ('Dataset Creation', self.step2_create_datasets),
                ('Model Initialization', self.step3_initialize_models),
                ('Model Training', self.step4_train_models),
                ('Model Evaluation', self.step5_evaluate_models),
                ('Multi-Step Forecasting', self.step6_multi_step_forecast)
            ]

            for step_name, step_func in steps:
                if not step_func():
                    self.logger.error(f"Pipeline failed at step: {step_name}")
                    return False

            # Save final summary
            self.save_final_summary()

            # Log completion
            end_time = datetime.now()
            duration = end_time - start_time
            self.logger.info("\n" + "="*80)
            self.logger.info("Pipeline Completed Successfully!")
            self.logger.info("="*80)
            self.logger.info(f"Start time: {start_time}")
            self.logger.info(f"End time: {end_time}")
            self.logger.info(f"Duration: {duration}")
            self.logger.info(f"Output directory: {self.run_dir}")

            return True

        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            return False


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults.

    Args:
        config_file: Path to JSON configuration file

    Returns:
        Configuration dictionary
    """
    default_config = {
        'seed': 30,
        'use_macro': True,
        'output_dir': './output',
        'raw_data_file': '../raw_data.csv',
        'processed_data_file': 'df_monthly.csv',
        'train_ratio': 0.80,
        'val_ratio': 0.07,
        'lookback_window': 6,
        'pred_horizon': 6,
        'batch_size': 16,
        'model_params': {
            'hidden_dim': 64,
            'time_varying_decay': True,
            'cnn_out_channels': 32,
            'cnn_kernel_size': 3,
            'transformer_dim': 64,
            'nhead': 4,
            'num_transformer_layers': 2,
            'rnn_hidden_dim': 64,
            'num_rnn_layers': 2,
            'lstm_hidden_dim': 64,
            'num_lstm_layers': 2,
            'dropout': 0.1
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'n_epochs': 100,
            'patience': 30
        }
    }

    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            custom_config = json.load(f)
        # Merge with defaults
        default_config.update(custom_config)

    return default_config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='DLNS Yield Curve Modeling Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python main.py

  # Run without macro variables
  python main.py --no-macro

  # Run with custom seed
  python main.py --seed 42

  # Run with custom configuration file
  python main.py --config my_config.json
        """
    )

    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--use-macro', action='store_true', help='Use macroeconomic variables')
    parser.add_argument('--no-macro', action='store_true', help='Do not use macroeconomic variables')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--output-dir', type=str, help='Output directory path')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command-line arguments
    if args.no_macro:
        config['use_macro'] = False
    elif args.use_macro:
        config['use_macro'] = True

    if args.seed is not None:
        config['seed'] = args.seed

    if args.output_dir:
        config['output_dir'] = args.output_dir

    # Create and run pipeline
    pipeline = DLNSPipeline(config)
    success = pipeline.run()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
