import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x = np.array(x)
        eps = 1e-5
        mean = np.mean((x)**2)
        rms = np.sqrt(mean + eps)
        x_hat = x/rms
        output = gamma * x_hat
        return np.round(output,4)
