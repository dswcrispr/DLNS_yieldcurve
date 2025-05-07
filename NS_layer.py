import torch
import torch.nn as nn
import numpy as np

class NelsonSiegelLayer(nn.Module):
    """
    PyTorch implementation of the Nelson-Siegel yield curve model as a neural network layer.
    
    Supports both time-invariant and time-varying decay parameter λ:
    - Time-invariant: λ is determined only by a bias term
    - Time-varying: λ is an additional latent factor alongside level, slope, and curvature
    - The λ is always provided as input from the pre-NS layer, following the paper
    """
    
    def __init__(self, maturities, eps=1e-10):
        """
        Initialize the Nelson-Siegel layer.
        
        Args:
            maturities (list or numpy array): Tenors/maturities in years
            eps (float): Small constant to prevent division by zero
        """
        super(NelsonSiegelLayer, self).__init__()
        self.maturities = torch.tensor(maturities, dtype=torch.float32)
        self.eps = eps
        
    
    def constrain_lambda(self, lambda_raw):
        """
        Apply constraints to ensure λ is positive and within a reasonable range (around 24-60 months)
        Using sigmoid as in the paper (Equation 9).
        
        Args:
            lambda_raw (torch.Tensor): Raw (unconstrained) λ values
            
        Returns:
            torch.Tensor: Constrained λ values
        """
        # From Equation 9: λ = Sigmoid(λ̃) × (1.8/24 - 1.8/60) + 1.8/60
        min_val = 1.8/60  # λ for maximal loading at 60 months
        max_val = 1.8/24  # λ for maximal loading at 24 months
        scale = max_val - min_val
        
        return torch.sigmoid(lambda_raw) * scale + min_val
    
    def get_factor_loadings(self, decay):
        """
        Calculate the Nelson-Siegel factor loadings for the given maturities.
        
        Args:
            decay (torch.Tensor): Decay parameter λ, shape depends on batch size
            
        Returns:
            tuple: (level_loading, slope_loading, curvature_loading)
        """
        # Ensure decay has the right shape for broadcasting
        if decay.dim() == 1:
            decay = decay.unsqueeze(-1)  # [batch_size, 1]
        
        # Prepare maturities tensor for broadcasting
        maturities = self.maturities.to(decay.device)
        
        # Prevent division by zero for very short maturities
        tau = torch.clamp(maturities, min=self.eps)
        
        # Calculate factor loadings
        # Level factor loading is constant at 1
        level_loading = torch.ones_like(tau)
        
        # Prepare for broadcasting with batch dimension
        # Expand tau to match batch size if needed
        if decay.dim() > 1 and decay.size(0) > 1:
            # Make sure tau has correct dimensions for broadcasting
            tau = tau.reshape(1, -1).expand(decay.size(0), -1)
            level_loading = level_loading.reshape(1, -1).expand(decay.size(0), -1)
        
        # Calculate exponential term
        decay_tau = decay * tau  # [batch_size, n_maturities] or [1, n_maturities]
        exp_term = torch.exp(-decay_tau)
        
        # Calculate common term (1-e^(-λτ))/(λτ)
        common_term = (1 - exp_term) / decay_tau
        
        # Slope factor loading
        slope_loading = common_term
        
        # Curvature factor loading
        curvature_loading = common_term - exp_term
        
        return level_loading, slope_loading, curvature_loading
    
    def forward(self, factors):
        """
        Forward pass to compute yields using Nelson-Siegel factors.
        
        Args:
            factors (torch.Tensor): Tensor containing Nelson-Siegel factors
                                   - Always shape [batch_size, 4] (level, slope, curvature, λ_raw) 
        
        Returns:
            torch.Tensor: Yields for each maturity, shape [batch_size, len(maturities)]
        """
        batch_size = factors.shape[0]

        # Extract raw λ from factors and apply constraint
        lambda_raw = factors[:, 3]  # [batch_size]
        decay = self.constrain_lambda(lambda_raw)
        
        # Get factor loadings
        level_loading, slope_loading, curvature_loading = self.get_factor_loadings(decay)
        
        # Extract level, slope, and curvature factors and ensure they have the correct shape
        # convert to [batch_size, 1] for proper broadcasting
        level = factors[:, 0].reshape(batch_size, 1)  # [batch_size, 1]
        slope = factors[:, 1].reshape(batch_size, 1)  # [batch_size, 1]
        curvature = factors[:, 2].reshape(batch_size, 1)  # [batch_size, 1]
                
        # Calculate yields for each maturity
        yields = (level * level_loading + 
                 slope * slope_loading + 
                 curvature * curvature_loading)
        
        return yields, decay

# -------------------Debug test--------------------------------
# Debug test of NS layer
import torch

# Create a small test
maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
ns_layer = NelsonSiegelLayer(maturities)

# Create fake factors
batch_size = 32
factors = torch.randn(batch_size, 4)  # level, slope, curvature, lambda

# Forward pass
yields, decay = ns_layer(factors)

print(f"Input factors shape: {factors.shape}")
print(f"Output yields shape: {yields.shape}")
print(f"Lambda shape: {decay.shape}")
