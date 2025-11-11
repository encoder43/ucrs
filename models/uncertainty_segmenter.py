"""
Uncertainty-Guided Probabilistic Segmenter (ObjSeeker)
Implements Proposal 1: Uncertainty-Guided Probabilistic Slicing

This module extends the Segmenter to output both mean objectness and uncertainty (variance),
enabling probabilistic patch selection that prioritizes high-confidence regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UncertaintySegmenter(nn.Module):
    """
    Probabilistic ObjSeeker that predicts mean objectness and uncertainty.
    
    Outputs:
        - mean_objectness: Mean objectness map M_μ ∈ R^(H/8 × W/8)
        - uncertainty: Uncertainty map M_σ² ∈ R^(H/8 × W/8)
    
    The uncertainty is modeled using:
    1. Evidential learning (recommended): Direct variance prediction
    2. Monte Carlo Dropout: Variance from multiple forward passes
    """
    
    def __init__(self, nc=1, ch=(), method='evidential', dropout_rate=0.1):
        """
        Args:
            nc: Number of output channels (usually 1 for binary objectness)
            ch: Input channels from backbone
            method: 'evidential' or 'dropout'
            dropout_rate: Dropout rate for Monte Carlo method
        """
        super(UncertaintySegmenter, self).__init__()
        self.nc = nc
        self.method = method
        
        if method == 'evidential':
            # Evidential approach: predict parameters of a distribution
            # Output 2 channels: α (concentration) and β (rate) for gamma distribution
            # Mean: α/β, Variance: α/β²
            self.m = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(x, x, 3, 1, 1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(x, 2 * nc, 1, 1, 0, bias=True)  # α, β for each class
                ) for x in ch
            ])
            # Initialize to have reasonable prior (low confidence initially)
            for m in self.m:
                if isinstance(m, nn.Sequential):
                    for layer in m:
                        if isinstance(layer, nn.Conv2d) and layer.kernel_size == (1, 1):
                            nn.init.constant_(layer.bias[:nc], 1.0)  # α = 1
                            nn.init.constant_(layer.bias[nc:], 0.5)   # β = 0.5
                            
        elif method == 'dropout':
            # Monte Carlo Dropout: Use dropout at inference
            self.m = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout2d(dropout_rate),
                    nn.Conv2d(x, x, 3, 1, 1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout_rate),
                    nn.Conv2d(x, nc, 1, 1, 0, bias=True)
                ) for x in ch
            ])
            self.dropout_rate = dropout_rate
            self._enable_dropout_at_inference = True
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'evidential' or 'dropout'")
    
    def forward(self, x, num_samples=1):
        """
        Forward pass
        
        Args:
            x: List of feature maps from backbone
            num_samples: Number of samples for Monte Carlo (only for dropout method)
        
        Returns:
            mean_objectness: Mean objectness map M_μ
            uncertainty: Uncertainty map M_σ² (variance)
        """
        if self.method == 'evidential':
            return self._forward_evidential(x)
        else:  # dropout
            return self._forward_dropout(x, num_samples)
    
    def _forward_evidential(self, x):
        """Evidential learning forward pass"""
        outputs = []
        for i, feat in enumerate(x):
            out = self.m[i](feat)  # [B, 2*nc, H, W]
            
            # Split into α and β
            alpha = F.softplus(out[:, :self.nc]) + 1.0  # Ensure α > 1
            beta = F.softplus(out[:, self.nc:]) + 1e-6  # Ensure β > 0
            
            # Compute mean and variance
            mean = alpha / beta
            variance = alpha / (beta ** 2)
            
            # Take mean across channels if nc > 1
            if self.nc > 1:
                mean = mean.mean(dim=1, keepdim=True)
                variance = variance.mean(dim=1, keepdim=True)
            
            outputs.append((mean, variance))
        
        # Use the finest scale (first output) for patch selection
        return outputs[0][0], outputs[0][1]
    
    def _forward_dropout(self, x, num_samples=1):
        """Monte Carlo Dropout forward pass"""
        if self.training or num_samples == 1:
            # Single forward pass during training
            outputs = [self.m[i](feat) for i, feat in enumerate(x)]
            mean = outputs[0].sigmoid()
            
            # During training, we don't have uncertainty from dropout
            # Use a learned uncertainty head or constant
            if hasattr(self, 'uncertainty_head'):
                uncertainty = F.softplus(self.uncertainty_head(x[0])) + 1e-6
            else:
                # Heuristic: uncertainty is inverse of confidence
                uncertainty = (1.0 - mean) * 0.1 + 1e-6
        else:
            # Multiple forward passes for Monte Carlo estimation
            samples = []
            for _ in range(num_samples):
                if not self._enable_dropout_at_inference:
                    # Temporarily enable dropout
                    for module in self.m[0]:
                        if isinstance(module, nn.Dropout2d):
                            module.train()
                
                out = self.m[0](x[0]).sigmoid()
                samples.append(out)
                
                if not self._enable_dropout_at_inference:
                    # Disable dropout
                    for module in self.m[0]:
                        if isinstance(module, nn.Dropout2d):
                            module.eval()
            
            # Compute mean and variance
            samples = torch.stack(samples, dim=0)  # [num_samples, B, C, H, W]
            mean = samples.mean(dim=0)
            variance = samples.var(dim=0)
            
            # Expand to list format for consistency
            outputs = [mean]
        
        return outputs[0], variance if not self.training else (1.0 - mean) * 0.1 + 1e-6
    
    def set_mc_inference(self, enable=True, num_samples=10):
        """Enable/disable Monte Carlo dropout at inference"""
        self._enable_dropout_at_inference = enable
        self._num_mc_samples = num_samples


def compute_uncertainty_loss(mean_pred, var_pred, target_mask, reduction='mean'):
    """
    Compute uncertainty-aware loss for ObjSeeker training
    
    Uses Gaussian Negative Log-Likelihood (NLL) to learn both mean and variance
    
    Args:
        mean_pred: Predicted mean objectness M_μ
        var_pred: Predicted uncertainty (variance) M_σ²
        target_mask: Ground truth mask M
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        loss: Uncertainty-aware loss
    """
    # Ensure variance is positive
    var_pred = F.softplus(var_pred) + 1e-6
    
    # Gaussian NLL: -log N(target | mean, var)
    # = 0.5 * log(2π*var) + 0.5 * (target - mean)^2 / var
    log_var = torch.log(var_pred + 1e-6)
    mse_term = (target_mask - mean_pred) ** 2
    nll = 0.5 * log_var + 0.5 * mse_term / var_pred
    
    # Add focal-like weighting for difficult examples
    # Higher uncertainty regions get more weight in loss
    confidence_weight = 1.0 / (var_pred + 1e-6)
    confidence_weight = confidence_weight / confidence_weight.mean()  # Normalize
    
    weighted_nll = nll * confidence_weight
    
    if reduction == 'mean':
        return weighted_nll.mean()
    elif reduction == 'sum':
        return weighted_nll.sum()
    else:
        return weighted_nll

