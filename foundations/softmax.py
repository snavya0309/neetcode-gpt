import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maximum = max(z)
        e_z = np.exp(z-maximum)
        e_s = np.sum(e_z)
        result = e_z/e_s
        return np.round(result,4)


