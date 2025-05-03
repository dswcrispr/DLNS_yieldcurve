import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Transformer model.
    Adds information about the position of tokens in the sequence.
    """
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # unsqueeze(1) is to add a new dimension at index 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # torch.arange(0, d_model, 2) is to create a tensor of even indices
        
        # Sine for even indices, cosine for odd indices
        pe[:, 0::2] = torch.sin(position * div_term) # shape of pe is (max_len, d_model)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        # buffer is a special type of parameter that is not optimized by the optimizer
        self.register_buffer('pe', pe.unsqueeze(0)) # now, pe is (1, max_len, d_model)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)] # match the sequence length

class CNN1DTransformerEncoder(nn.Module):
    """
    Hybrid 1D CNN + Transformer encoder for yield curve forecasting.
    
    The architecture:
    1. 1D CNN extracts local features from each time step's yield curve
    2. Transformers process the sequence of extracted features to capture temporal dependencies
    """
    def __init__(
        self, 
        input_dim,                # Total input dimension (all yields + macro variables)
        yield_dim,                # Number of yield maturities
        macro_dim,                # Number of macro variables
        hidden_dim=64,            # Hidden dimension for final projection # check point, how is it related to 4 factors output from Pre-NS?
        cnn_out_channels=32,      # CNN output channels
        cnn_kernel_size=3,        # CNN kernel size
        transformer_dim=64,       # Transformer dimension, in small datasets(~500) 32~64 is enough
        nhead=4,                  # Number of attention heads, total transformer dimension should be divisible by nhead
        num_transformer_layers=2, # Number of transformer layers, too many layers will cause overfitting
        dropout=0.1,              # Dropout rate
        seq_length=12,            # Sequence length (lookback window)
    ):
        super(CNN1DTransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.yield_dim = yield_dim
        self.macro_dim = macro_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        # 1D CNN for extracting features from yield curves
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1,                     # 1D sequence of yields
                out_channels=cnn_out_channels,     # Output channels, 32
                kernel_size=cnn_kernel_size,       # Kernel size
                padding=(cnn_kernel_size-1)//2,    # Same padding
            ),
            nn.ELU(), # ELU is a type of activation function, it is a smooth version of ReLU
            nn.MaxPool1d(kernel_size=2, stride=2), # Reduce dimension, 1/2
            
            nn.Conv1d(
                in_channels=cnn_out_channels, 
                out_channels=cnn_out_channels*2,
                kernel_size=cnn_kernel_size,
                padding=(cnn_kernel_size-1)//2,
            ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Further reduce dimension, 1/4
        )
        
        # Calculate CNN output size based on yield_dim
        # Two MaxPool1d layers with stride 2 reduce dimension by factor of 4
        cnn_output_size = yield_dim // 4 * (cnn_out_channels*2) 
        
        # Linear layer to project CNN output to transformer dimension
        self.cnn_projector = nn.Linear(cnn_output_size, transformer_dim - macro_dim) # macro variables are used as additional features for the transformer
        
        # Positional encoding for transformer
        self.positional_encoding = PositionalEncoding(d_model=transformer_dim)
        
        # Transformer encoder
        # Define the layer of the transformer first
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, # input and output dimension of the transformer
            nhead=nhead,
            dim_feedforward=transformer_dim*4,
            dropout=dropout,
            batch_first=True
        )
        # Then, define the encoder
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, 
            num_layers=num_transformer_layers # number of layers, 2
        )
        
        # Final projection to get the hidden features
        self.final_projection = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim), # check point, should I use the smaller hidden_dim?
            nn.Tanh()
        )
        
    def forward(self, x):
        """
        Forward pass through the CNN1D + Transformer encoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
        
        Returns:
            Tensor of shape [batch_size, hidden_dim]
        """
        batch_size = x.shape[0]
        
        # Split yield data and macro data
        yield_data = x[:, :, :self.yield_dim]  # [batch_size, seq_length, yield_dim]
        macro_data = x[:, :, self.yield_dim:]  # [batch_size, seq_length, macro_dim]
        
        # Process yield data through CNN
        cnn_features = []
        for t in range(self.seq_length): # seq_length is 12, lookback window
            # Get yield curve at time step t
            yield_t = yield_data[:, t, :].unsqueeze(1)  # [batch_size, 1, yield_dim]
            
            # Apply CNN
            cnn_out = self.cnn(yield_t)  # [batch_size, cnn_out_channels*2, yield_dim//4]
            
            # Flatten and project
            cnn_out_flat = cnn_out.reshape(batch_size, -1)  # [batch_size, cnn_out_channels*2 * yield_dim//4]
            cnn_projected = self.cnn_projector(cnn_out_flat)  # [batch_size, transformer_dim - macro_dim]
            
            cnn_features.append(cnn_projected)
        
        # Stack CNN features
        cnn_features = torch.stack(cnn_features, dim=1)  # [batch_size, seq_length, transformer_dim - macro_dim]
        
        # Concatenate with macro data for each time step
        combined_features = torch.cat([cnn_features, macro_data], dim=2)  # [batch_size, seq_length, transformer_dim]
        
        # Add positional encoding
        combined_features = self.positional_encoding(combined_features) # this calls forward function of PositionalEncoding class automatically
        
        # Process through transformer
        transformer_output = self.transformer_encoder(combined_features) # this calls forward method of TransformerEncoder class automatically
        
        # Use the final sequence representation (last token)
        final_representation = transformer_output[:, -1, :]  # [batch_size, transformer_dim]
        
        # Project to hidden dimension
        hidden_features = self.final_projection(final_representation)  # [batch_size, hidden_dim]
        
        return hidden_features

class TransformerOnlyEncoder(nn.Module):
    """
    Transformer-only encoder for yield curve forecasting.
    
    The architecture processes the entire yield curve and macroeconomic variables 
    as a unified sequence, allowing the self-attention mechanism to learn dependencies
    between different maturities and macroeconomic indicators.
    """
    def __init__(
        self, 
        input_dim,                # Total input dimension (all yields + macro variables)
        hidden_dim=64,            # Hidden dimension for final projection
        transformer_dim=64,       # Transformer dimension
        nhead=4,                  # Number of attention heads
        num_transformer_layers=2, # Number of transformer layers
        dropout=0.1,              # Dropout rate
        seq_length=12,            # Sequence length (lookback window)
    ):
        super(TransformerOnlyEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        # Linear embedding to project input to transformer dimension
        self.input_projection = nn.Linear(input_dim, transformer_dim)
        
        # Positional encoding for transformer
        self.positional_encoding = PositionalEncoding(d_model=transformer_dim)
        
        # Transformer encoder
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=nhead,
            dim_feedforward=transformer_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # Final projection to get the hidden features
        self.final_projection = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        """
        Forward pass through the Transformer-only encoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
        
        Returns:
            Tensor of shape [batch_size, hidden_dim]
        """
        batch_size = x.shape[0]
        
        # Project input to transformer dimension
        projected_input = self.input_projection(x)  # [batch_size, seq_length, transformer_dim]
        
        # Add positional encoding
        encoded_input = self.positional_encoding(projected_input)
        
        # Process through transformer
        transformer_output = self.transformer_encoder(encoded_input)
        
        # Use the final sequence representation (last token)
        final_representation = transformer_output[:, -1, :]  # [batch_size, transformer_dim]
        
        # Project to hidden dimension
        hidden_features = self.final_projection(final_representation)  # [batch_size, hidden_dim]
        
        return hidden_features