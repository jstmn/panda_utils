from typing import Any
import torch
import numpy as np
import h5py


def cm_to_m(cm: float) -> float:
    return cm / 100.0


def to_public_dict(obj):
    """Recursively convert an object to a dict, skipping private attributes."""
    # Base cases: primitives
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_public_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: to_public_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: to_public_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if hasattr(obj, "__slots__"):
        return {
            slot: to_public_dict(getattr(obj, slot))
            for slot in obj.__slots__
            if hasattr(obj, slot) and not slot.startswith("_")
        }
    # Fallback to dir()
    result = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        value = getattr(obj, attr)
        if callable(value):
            continue
        result[attr] = to_public_dict(value)
    return result


def assert_is_4x4_matrix(matrix: np.ndarray):
    """Assert that the matrix is a 4x4 matrix"""
    assert matrix.shape == (4, 4), f"Expected shape (4, 4), got {matrix.shape}"
    assert np.isclose(
        np.linalg.det(matrix), 1.0, rtol=0.0, atol=0.001
    ), f"The determinant of the matrix should be 1.0, got {np.linalg.det(matrix)}"


def print_h5py_contents(obj, indent=0):
    """Print the contents of an h5py object recursively"""
    for key in obj.keys():
        item = obj[key]
        print(" " * indent + f"/{key}: ", end="")
        if isinstance(item, h5py.Dataset):
            print(f"Dataset {item.shape} {item.dtype}")
        elif isinstance(item, h5py.Group):
            print("Group")
            print_h5py_contents(item, indent + 4)
        elif isinstance(item, torch.Tensor):
            print(f"Tensor {item.shape} {item.dtype}")
        else:
            print(f"Unknown type: {type(item)}")


def print_variable_recursively(d: Any, indent=0, max_items=10):
    """Print the contents of a variable recursively with better formatting and type handling"""
    indent_str = "  " * indent

    if isinstance(d, torch.Tensor):
        print(f"{indent_str}torch.tensor: shape={d.shape}, dtype={d.dtype}, device={d.device}")
        if d.numel() <= 20:  # Show values for small tensors
            print(f"{indent_str}  values: {d.flatten().tolist()}")
        else:
            # Convert to float for statistical operations if needed
            d_float = d.float() if d.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64) else d
            print(
                f"{indent_str}  min={d_float.min().item():.4f}, max={d_float.max().item():.4f}, mean={d_float.mean().item():.4f}"
            )

    elif isinstance(d, np.ndarray):
        print(f"{indent_str}numpy.ndarray: shape={d.shape}, dtype={d.dtype}")
        if d.size <= 20:  # Show values for small arrays
            print(f"{indent_str}  values: {d.flatten().tolist()}")
        else:
            print(f"{indent_str}  min={d.min():.4f}, max={d.max():.4f}, mean={d.mean():.4f}")

    elif isinstance(d, dict):
        print(f"{indent_str}dict with {len(d)} keys:")
        for i, (key, value) in enumerate(d.items()):
            if i >= max_items:
                print(f"{indent_str}  ... and {len(d) - max_items} more items")
                break
            print(f"{indent_str}  - {key}: ", end="")
            print_variable_recursively(value, indent + 4, max_items)

    elif isinstance(d, list):
        print(f"{indent_str}list with {len(d)} items:")
        for i, item in enumerate(d):
            if i >= max_items:
                print(f"{indent_str}  ... and {len(d) - max_items} more items")
                break
            print(f"{indent_str}  [{i}]: ", end="")
            print_variable_recursively(item, indent + 4, max_items)

    elif isinstance(d, tuple):
        print(f"{indent_str}tuple with {len(d)} items:")
        for i, item in enumerate(d):
            if i >= max_items:
                print(f"{indent_str}  ... and {len(d) - max_items} more items")
                break
            print(f"{indent_str}  ({i}): ", end="")
            print_variable_recursively(item, indent + 4, max_items)

    elif isinstance(d, (int, float, str, bool)):
        print(f"{indent_str}{d} ({type(d).__name__})")

    elif d is None:
        print(f"{indent_str}None")

    else:
        print(f"{indent_str}{type(d).__name__}: {str(d)[:100]}{'...' if len(str(d)) > 100 else ''}")
