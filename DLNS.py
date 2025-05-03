
import torch
import torch.nn as nn
import numpy as np
from Encoder import CNN1DTransformerEncoder, TransformerOnlyEncoder
from NS_layer import NelsonSiegelLayer

class DLNS_CNNTransformer(nn.Module):
    def __init__(
        self, 
        input_dim,                # Total input dimension (all yields + macro variables)
        yield_dim,                # Number of yield maturities
        macro_dim,                # Number of macro variables
        maturities,               # Maturities for yield prediction
        hidden_dim=64,            # Hidden dimension
        time_varying_decay=False, # Whether to use time-varying decay parameter
        cnn_out_channels=32,      # CNN output channels
        cnn_kernel_size=3,        # CNN kernel size
        transformer_dim=64,       # Transformer dimension
        nhead=4,                  # Number of attention heads
        num_transformer_layers=2, # Number of transformer layers
        dropout=0.1,              # Dropout rate
        seq_length=12,            # Sequence length (lookback window)
    ):
        super(DLNS_CNNTransformer, self).__init__()
        
        self.time_varying_decay = time_varying_decay
        
        # CNN + Transformer encoder
        self.encoder = CNN1DTransformerEncoder(
            input_dim=input_dim,
            yield_dim=yield_dim,
            macro_dim=macro_dim,
            hidden_dim=hidden_dim,
            cnn_out_channels=cnn_out_channels,
            cnn_kernel_size=cnn_kernel_size,
            transformer_dim=transformer_dim,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
            seq_length=seq_length,
        )
        
        # Always predict 4 factors (level, slope, curvature, lambda) regardless of time-varying decay
        if time_varying_decay:
            # For time-varying λ, predict all factors including λ directly
            self.factor_predictor = nn.Linear(hidden_dim, 4)
        else:
            # For time-invariant λ, implement Equation 12 structure in key paper with zero constraints
            # Separate layers for level, slope, and curvature
            self.level_predictor = nn.Linear(hidden_dim, 1)
            self.slope_predictor = nn.Linear(hidden_dim, 1)
            self.curvature_predictor = nn.Linear(hidden_dim, 1)
            
            # For λ, create a bias parameter with zero weights
            # This correctly implements Equation 12 from the paper, check point: appropriate initial value?
            self.lambda_bias = nn.Parameter(torch.tensor(1.8/36, dtype=torch.float32))
        
        # Nelson-Siegel layer
        self.ns_layer = NelsonSiegelLayer(maturities)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input features, shape [batch_size, seq_length, input_dim]
            
        Returns:
            tuple: (yields, factors, lambda_value)
                - yields: predicted yields, shape [batch_size, len(maturities)]
                - factors: NS factors, shape [batch_size, 4]
                - lambda_value: constrained decay parameter value, shape [batch_size]
        """
        batch_size = x.shape[0]
        
        # Extract features from encoder
        features = self.encoder(x)
        
        # Predict NS factors
        if self.time_varying_decay:
            # For time-varying λ, predict all factors including raw λ
            factors = self.factor_predictor(features)
        else:
            # For time-invariant λ, implement Equation 12 structure
            level = self.level_predictor(features)
            slope = self.slope_predictor(features)
            curvature = self.curvature_predictor(features)

            # For λ, use the learned bias parameter
            # This is time-invariant as it doesn't depned on the input features
            lambda_raw = self.lambda_bias.expand(batch_size, 1)
            
            # Combine the four factors 
            factors = torch.cat([level, slope, curvature, lambda_raw], dim=1)
        
        # Use NS layer to generate yields
        yields, lambda_value = self.ns_layer(factors)
        
        return yields, factors, lambda_value

class DLNSTransformerOnly(nn.Module):
    def __init__(
        self, 
        input_dim,                # Total input dimension (all yields + macro variables)
        maturities,               # Maturities for yield prediction
        hidden_dim=64,            # Hidden dimension
        time_varying_decay=False, # Whether to use time-varying decay parameter
        transformer_dim=64,       # Transformer dimension
        nhead=4,                  # Number of attention heads
        num_transformer_layers=2, # Number of transformer layers
        dropout=0.1,              # Dropout rate
        seq_length=12,            # Sequence length (lookback window)
    ):
        super(DLNSTransformerOnly, self).__init__()
        
        self.time_varying_decay = time_varying_decay
        
        # Transformer-only encoder
        self.encoder = TransformerOnlyEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            transformer_dim=transformer_dim,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
            seq_length=seq_length,
        )
        
        # Always predict 4 factors (level, slope, curvature, lambda)
        if time_varying_decay:
            # For time-varying λ, predict all factors including λ directly
            self.factor_predictor = nn.Linear(hidden_dim, 4)
        else:
            # For time-invariant λ, implement Equation 12 structure with zero constraints
            # Separate layers for level, slope, and curvature
            self.level_predictor = nn.Linear(hidden_dim, 1)
            self.slope_predictor = nn.Linear(hidden_dim, 1)
            self.curvature_predictor = nn.Linear(hidden_dim, 1)
            
            # For λ, create a bias parameter with zero weights
            # This correctly implements Equation 12 from the paper, check point: appropriate initial value?
            self.lambda_bias = nn.Parameter(torch.tensor(1.8/36, dtype=torch.float32))
        
        # Nelson-Siegel layer
        self.ns_layer = NelsonSiegelLayer(maturities)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input features, shape [batch_size, seq_length, input_dim]
            
        Returns:
            tuple: (yields, factors, lambda_value)
                - yields: predicted yields, shape [batch_size, len(maturities)]
                - factors: NS factors, shape [batch_size, 3] or [batch_size, 4]
                - lambda_value: decay parameter value, shape [batch_size]
        """
        batch_size = x.shape[0]
        
        # Extract features from encoder
        features = self.encoder(x)
        
        # Predict NS factors
        if self.time_varying_decay:
            # For time-varying λ, predict all factors including raw λ
            factors = self.factor_predictor(features)
        else:
            # For time-invariant λ, implement Equation 12 structure
            level = self.level_predictor(features)
            slope = self.slope_predictor(features)
            curvature = self.curvature_predictor(features)

            # For λ, use the learned bias parameter
            # This is time-invariant as it doesn't depned on the input features
            lambda_raw = self.lambda_bias.expand(batch_size, 1)
            
            # Combine the four factors
            factors = torch.cat([level, slope, curvature, lambda_raw], dim=1)
        
        # Use NS layer to generate yields
        yields, lambda_value = self.ns_layer(factors)
        
        return yields, factors, lambda_value