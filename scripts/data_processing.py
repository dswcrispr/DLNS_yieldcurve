"""
Data Processing Script for Yield Curve Analysis

This script processes yield curve data by loading, preprocessing, and preparing it
for model training. It uses the data_preprocessing module to handle the raw data.
"""

import pandas as pd
import sys
import os
from typing import Tuple
import warnings

# Add modules directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
from modules.data_preprocessing import process_yield_curve_data

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)


def load_and_process_data(csv_file: str, output_file: str = 'df_monthly.csv') -> pd.DataFrame:
    """
    Load and process yield curve data from CSV file.

    Args:
        csv_file: Path to the raw yield curve CSV file
        output_file: Path to save the processed monthly data

    Returns:
        Processed monthly DataFrame with yield curves and macro variables
    """
    df_monthly = process_yield_curve_data(csv_file)
    return df_monthly


def verify_processed_data(csv_file: str = 'df_monthly.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and verify the processed data by displaying head and tail.

    Args:
        csv_file: Path to the processed monthly data CSV file

    Returns:
        Tuple of (head DataFrame, tail DataFrame) for verification
    """
    df = pd.read_csv(csv_file, index_col=0)

    print("First few rows of processed data:")
    print(df.head())
    print("\nLast few rows of processed data:")
    print(df.tail())

    return df.head(), df.tail()


def main() -> None:
    """
    Main execution function for data processing workflow.
    """
    # Define input CSV file path
    csv_file = '../raw_data.csv'

    # Process the yield curve data
    print("Processing yield curve data...")
    df_monthly = load_and_process_data(csv_file)
    print(f"Data processed successfully. Shape: {df_monthly.shape}")

    # Verify the processed data
    print("\nVerifying processed data...")
    verify_processed_data('df_monthly.csv')

    print("\nData processing complete!")


if __name__ == '__main__':
    main()
