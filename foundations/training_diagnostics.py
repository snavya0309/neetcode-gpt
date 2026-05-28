import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np

class Solution:
    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        stats = []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.Linear):
                    # Added .item() to ensure pure Python floats are returned
                    mean_val = round(x.mean().item(), 4)
                    std_val = round(x.std().item(), 4)
                    
                    if x.dim() >= 2:
                        dead_frac = round(((x <= 0).all(dim=0)).float().mean().item(), 4)
                    else:
                        dead_frac = round((x <= 0).float().mean().item(), 4)
                        
                    stats.append({'mean': mean_val, 'std': std_val, 'dead_fraction': dead_frac})
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        stats = []
        # FIXED: Added () to children call
        for module in model.children():
            # FIXED: Checked module, not model
            if isinstance(module, nn.Linear):
                # FIXED: Grabbed grad from module, not model
                grad = module.weight.grad
                
                # Added .item() to ensure pure Python floats are returned
                stats.append({
                    'mean': round(grad.mean().item(), 4),
                    'std': round(grad.std().item(), 4),
                    'norm': round(torch.norm(grad).item(), 4)
                })
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        for s in activation_stats:
            if s['dead_fraction'] > 0.5:
                return 'dead_neurons'
                
        for s in gradient_stats:
            if s['norm'] > 1000:
                return 'exploding_gradients'
                
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
            
        for s in activation_stats:
            if s['std'] < 0.1:
                return 'vanishing_gradients'
            if s['std'] > 10.0:
                return 'exploding_gradients'
                
        return 'healthy'
