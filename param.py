from typing import Tuple, Union

import numpy as np

Iterable = Union[list[int], list[float], list[str], tuple[int, ...], tuple[float, ...], np.ndarray]

#: Type alias for parameters that can be float, int, or numpy array
Param = Union[int, float, str, Iterable]


def to_array(value: Param) -> np.ndarray:
    """
    Convert a parameter to a numpy array if it is not already one.

    - Numeric types get cast to float64 arrays
    - Strings (or lists/tuples of strings) get inferred by numpy (usually <U... or object)
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (int, float)):
        return np.array(value, dtype=np.float64)
    return np.array(value)


def broadcast(*args: Param) -> Tuple[np.ndarray, ...]:
    """
    Normalize inputs to broadcast-compatible NumPy arrays using to_array, preserving order.

    Raises:
        ValueError: if broadcasting fails due to incompatible shapes.
    """
    arrays = [to_array(arg) for arg in args]
    try:
        return tuple(np.broadcast_arrays(*arrays))
    except ValueError as e:
        raise ValueError(
            f"Incompatible shapes for broadcasting: {[arr.shape for arr in arrays]}"
        ) from e
