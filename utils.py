import numpy as np
from numpy.typing import NDArray


def img_size(arr: NDArray):
    return np.array([arr.shape[1], arr.shape[0]])


def clamp(n, minn, maxn):
    if n < minn:
        return minn
    elif n > maxn:
        return maxn
    else:
        return n
