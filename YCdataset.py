import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class YieldCurveDataset(Dataset):
    def __init__(self, data, seq_length=12, pred_horizon=6, yield_scaler=None,
                 macro_scaler=None, is_train=True, use_macro=True):
        """
        Initialize the YieldCurveDataset class using sklearn scalers.

        Args:
            data (pd.DataFrame): Yield curve dataframe with dates as index and maturities/macro vars as columns
            seq_length (int): Number of past months to use as features
            pred_horizon (int): How many months ahead to predict (1-6 months)
            yield_scaler (MinMaxScaler): Pre-fitted scaler for yield data
            macro_scaler (StandardScaler): Pre-fitted scaler for macro variables
            is_train (bool): Whether this dataset is for training
            use_macro (bool): Whether to include macro variables as features
        """
        self.data = data.copy()
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.is_train = is_train
        self.use_macro = use_macro

        # Separate yield data from macro variables
        self.yield_cols = [col for col in data.columns if col not in ['NASDAQ', 'IP', 'CS']]
        self.macro_cols = ['NASDAQ', 'IP', 'CS'] if use_macro else None

        # Create or use provided scalers
        self.yield_scaler = yield_scaler
        self.macro_scaler = macro_scaler if use_macro else []

        # Normalize data
        self.normalized_data = self._normalize_data(data)

        # Prepare sequences (check point, what is sequecnes for?)
        self.sequences = [] # check point, pred_horizon=6 is valid?
        for i in range(len(self.normalized_data) - seq_length - pred_horizon + 1):
            # Only include sequences that can have full prediction horizon
            self.sequences.append(i)

    def _normalize_data(self, data):
        """
        Normalize the data using sklearn scalers.

        For yields: MinMaxScaler across all maturities
        For macro variables: StandardScaler
        """
        normalized_data = data.copy()

        # Handle yield data normalization
        if self.yield_scaler is None:
            if not self.is_train:
                raise ValueError("yield_scaler must be provided for validation/test datasets")
            self.yield_scaler = MinMaxScaler()
            yield_values = data[self.yield_cols].values
            # reshape to 2D array to scale across all maturities
            self.yield_scaler.fit(yield_values.reshape(-1, 1))

        # Handle macro data normalization (only if use_macro is True)
        if self.use_macro and self.macro_scaler is None:
            if not self.is_train:
                raise ValueError("macro_scaler must be provided for validation/test datasets")
            if self.macro_cols:  # Only if macro columns exist
                self.macro_scaler = StandardScaler()
                self.macro_scaler.fit(data[self.macro_cols]) # applied to each macro variable

        # Apply normalization to yield data
        yield_normalized = self.yield_scaler.transform(
            data[self.yield_cols].values.reshape(-1, 1)
        ).reshape(data.shape[0], len(self.yield_cols))
        normalized_data[self.yield_cols] = yield_normalized

        # Apply normalization to macro data if it exists
        if self.use_macro and self.macro_cols:
            macro_normalized = self.macro_scaler.transform(data[self.macro_cols])
            normalized_data[self.macro_cols] = macro_normalized

        # Create a new DataFrame with only the columns we need
        columns_to_keep = self.yield_cols + self.macro_cols
        normalized_data = normalized_data[columns_to_keep]

        return normalized_data

    def __len__(self): # Dataloader from PyTorch uses this function to get the length of the dataset
        return len(self.sequences) # this length means the total number of sequences can be used for training and validation

    def __getitem__(self, idx):
        seq_idx = self.sequences[idx]

        # Get input features (past seq_length months)
        X = self.normalized_data.iloc[seq_idx:seq_idx + self.seq_length].values # shape of X is (seq_length, num_features)

        # Get target values (next pred_horizon months)
        y = np.array([
            self.normalized_data.iloc[seq_idx + self.seq_length + horizon][self.yield_cols].values
            for horizon in range(self.pred_horizon)
        ])

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def denormalize_yields(self, normalized_yields):
        """
        Convert normalized yield values back to original scale using the scaler
        """
        # Reshape to 2D array for inverse_transform
        reshaped_yields = normalized_yields.reshape(-1, 1)
        denormalized = self.yield_scaler.inverse_transform(reshaped_yields)

        # Reshape back to original shape
        return denormalized.reshape(normalized_yields.shape)


## ----------- Use case in .ipynb ---------------    

# Data loading
# Load the processed monthly data from the CSV file, using 'Date' as the index
df = pd.read_csv('df_monthly.csv', index_col='Date')

# Define train/val size
# Calculate the size of the training and validation datasets
train_size = int(0.765 * len(df))  # 76.5% of the data for training
val_size = int(0.153 * len(df))    # 15.3% of the data for validation

# Load YieldCurveDataset Class
from YCdataset import YieldCurveDataset

# Create train dataset and get the scalers
# Initialize the training dataset and fit the scalers
train_dataset = YieldCurveDataset(df.iloc[:train_size], seq_length=12, pred_horizon=6, is_train=True, use_macro=True)
yield_scaler = train_dataset.yield_scaler  # Extract the fitted yield scaler
macro_scaler = train_dataset.macro_scaler  # Extract the fitted macro scaler

# Create validation and test datasets using the scalers from training
# Initialize the validation dataset using the scalers from the training dataset
val_dataset = YieldCurveDataset(
    df.iloc[train_size:train_size+val_size],
    seq_length=12,
    pred_horizon=6,
    yield_scaler=yield_scaler,
    macro_scaler=macro_scaler,
    is_train=False,
    use_macro=True
)

# Initialize the test dataset using the scalers from the training dataset
test_dataset = YieldCurveDataset(
    df.iloc[train_size+val_size:],
    seq_length=12,
    pred_horizon=6,
    yield_scaler=yield_scaler,
    macro_scaler=macro_scaler,
    is_train=False,
    use_macro=True
)

# The scalers can also be easily saved for future use:
'''
import joblib

# Save scalers
joblib.dump(yield_scaler, 'yield_scaler.pkl)
joblib.dump(macro_scaler, 'macro_scaler.pkl)

# Load scalers Later
yield_scaler = joblib.load('yield_scaler.pkl')
macro_scaler = joblib.load('macro_scaler.pkl')
'''