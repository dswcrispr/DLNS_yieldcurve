import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
from datetime import datetime
import os
import warnings

def process_yield_curve_data(csv_filename):
    """
    Process smoothed US Treasury yield curve data and FRED data.

    Parameters:
    csv_filename (str): The filename of the CSV file containing yield curve data.
    Here, we use data from
    'https://www.federalreserve.gov/econres/feds/the-us-treasury-yield-curve-1961-to-the-present.htm'

    Returns:
    pd.DataFrame: A DataFrame containing the processed monthly data.
    """
    # Suppress warnings and set pandas options for better readability
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', None)
    pd.set_option('mode.chained_assignment', None)

    # Load smoothed US Treasury yield curve data from a CSV file
    df = pd.read_csv(csv_filename)

    # Convert the 'Date' column to datetime format and standardize it to 'yyyy-mm-dd'
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    # Filter to keep only zero-coupon yield data columns
    maturities = ['SVENY' + str(i).zfill(2) for i in range(1, 31)]
    columns_to_keep = ['Date'] + [col for col in maturities if col in df.columns]
    df = df[columns_to_keep]

    # Rename columns to a more readable format
    df.columns = ['Date'] + [f'{i}Y' for i in range(1, 31)]

    # Forward fill NaN values and drop any remaining NaNs
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    # Set 'Date' as the index and ensure it's in datetime format
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)

    # Define the date range for fetching FRED data
    start_date = '1961-01-01'
    end_date = datetime.today().date()

    # Define symbols for daily and monthly FRED data
    symbols_d = {'3M': 'DGS3MO', '6M': 'DGS6MO', 'NASDAQ': 'NASDAQCOM'}
    symbols_m = {'IP': 'INDPRO', 'CS': 'UMCSENT'}

    # Initialize empty DataFrames
    fred_d = pd.DataFrame()
    fred_m = pd.DataFrame()

    # Fetch daily data from FRED
    for label, symbol in symbols_d.items():
        try:
            data = pdr.DataReader(symbol, 'fred', start_date, end_date)
            fred_d[label] = data
        except Exception as e:
            print(f"Error fetching data for {label}: {e}")

    # Fetch monthly data from FRED
    for label, symbol in symbols_m.items():
        try:
            data = pdr.DataReader(symbol, 'fred', start_date, end_date)
            fred_m[label] = data
        except Exception as e:
            print(f"Error fetching data for {label}: {e}")

    # Forward fill and drop NaNs for FRED data
    fred_d.fillna(method='ffill', inplace=True)
    fred_d.dropna(inplace=True)

    # Ensure the index is a DatetimeIndex
    fred_m.index = pd.to_datetime(fred_m.index)

    # Apply strftime to format the index
    fred_m.index = fred_m.index.strftime('%Y-%m')
    fred_m.dropna(inplace=True)

    # Merge yield curve data with daily FRED data
    df_daily = pd.merge(df, fred_d, how='left', left_index=True, right_index=True)

    # Reorder columns to place 3M and 6M yields at the start and NASDAQ at the end
    new_columns = ['3M', '6M'] + df.columns.tolist() + ['NASDAQ']
    df_daily = df_daily.reindex(columns=new_columns)

    # Extract yield columns and create a 'year_month' column for grouping
    yield_columns = [col for col in df_daily.columns if col != 'NASDAQ']
    df_daily['year_month'] = df_daily.index.strftime('%Y-%m')

    # Aggregate data to monthly frequency
    df_monthly_yields = df_daily.groupby('year_month')[yield_columns].last()
    df_monthly_stock = df_daily.groupby('year_month')['NASDAQ'].mean()

    # Combine monthly yield and stock data
    df_monthly = df_monthly_yields.copy()
    df_monthly['NASDAQ'] = df_monthly_stock

    # Merge with monthly FRED data and drop NaNs
    df_monthly = pd.merge(df_monthly, fred_m, how='left', left_index=True, right_index=True)
    df_monthly.dropna(inplace=True)

    # Rename index to 'Date'
    df_monthly.index.name = 'Date'

    # Apply log transformation and first difference to NASDAQ
    df_monthly['NASDAQ'] = np.log(df_monthly['NASDAQ']).diff()

    # Apply log transformation to IP and CS columns
    df_monthly['IP'] = np.log(df_monthly['IP'])
    df_monthly['CS'] = np.log(df_monthly['CS'])
    df_monthly.dropna(inplace=True)

    # Return the processed monthly data
    df_monthly.to_csv('df_monthly.csv', index=True)
    return df_monthly

# Example usage:
# df_monthly = process_yield_curve_data('feds200628.csv')
