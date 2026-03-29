"""
Neural Network Surrogate Model
================================
PyTorch MLP that predicts aerodynamic coefficients (CL, CD, CM)
from CST parameters and operating conditions (Mach, alpha).
"""

import torch
import torch.nn as nn
import numpy as np


class AirfoilSurrogate(nn.Module):
    """
    MLP surrogate model for airfoil aerodynamic prediction.
    
    Input:  [cst_upper_0..5, cst_lower_0..5, mach, alpha] = 14 features
    Output: [CL, CD, CM] = 3 targets
    """
    
    def __init__(self, input_dim=14, output_dim=3, hidden_dims=None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


class AirfoilDataset(torch.utils.data.Dataset):
    """Dataset for airfoil surrogate model training."""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class DataNormalizer:
    """Min-max normalization for inputs and outputs."""
    
    def __init__(self):
        self.feature_min = None
        self.feature_max = None
        self.target_min = None
        self.target_max = None
    
    def fit(self, features, targets):
        self.feature_min = features.min(axis=0)
        self.feature_max = features.max(axis=0)
        self.target_min = targets.min(axis=0)
        self.target_max = targets.max(axis=0)
        
        # Avoid division by zero
        feat_range = self.feature_max - self.feature_min
        feat_range[feat_range < 1e-10] = 1.0
        self.feature_range = feat_range
        
        tgt_range = self.target_max - self.target_min
        tgt_range[tgt_range < 1e-10] = 1.0
        self.target_range = tgt_range
    
    def transform_features(self, features):
        return (features - self.feature_min) / self.feature_range
    
    def transform_targets(self, targets):
        return (targets - self.target_min) / self.target_range
    
    def inverse_transform_targets(self, targets_norm):
        return targets_norm * self.target_range + self.target_min
    
    def state_dict(self):
        return {
            'feature_min': self.feature_min,
            'feature_max': self.feature_max,
            'feature_range': self.feature_range,
            'target_min': self.target_min,
            'target_max': self.target_max,
            'target_range': self.target_range
        }
    
    def load_state_dict(self, state):
        self.feature_min = state['feature_min']
        self.feature_max = state['feature_max']
        self.feature_range = state['feature_range']
        self.target_min = state['target_min']
        self.target_max = state['target_max']
        self.target_range = state['target_range']


def compute_r2(y_true, y_pred):
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return 1 - ss_res / (ss_tot + 1e-10)
