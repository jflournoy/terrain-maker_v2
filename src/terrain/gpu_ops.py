"""
GPU-accelerated operations using PyTorch.

This module provides GPU-accelerated versions of common terrain processing
operations. Functions automatically use CUDA when available, falling back
to CPU otherwise. All functions accept numpy arrays and return numpy arrays.

Key functions:
- gpu_horn_slope: Slope calculation using Horn's method (via conv2d)
- gpu_gaussian_blur: Gaussian blur (via F.conv2d with separable kernels)
"""

import numpy as np
import torch
import torch.nn.functional as F


def _get_device():
    """Get the best available device (CUDA > CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_horn_slope(dem: np.ndarray) -> np.ndarray:
    """
    Calculate slope magnitude using Horn's method with GPU acceleration.

    Uses PyTorch's F.conv2d for efficient convolution on GPU. Produces
    identical results to scipy.ndimage.convolve with Horn's kernels.

    Args:
        dem: 2D elevation data (H, W). Can contain NaN values.

    Returns:
        Slope magnitude array (same shape as input). NaN values preserved.
    """
    # Handle NaN values
    nan_mask = np.isnan(dem)
    has_nan = np.any(nan_mask)

    # Fill NaN values with interpolation for computation
    if has_nan:
        dem_filled = dem.copy()
        if np.any(~nan_mask):
            dem_filled[nan_mask] = np.interp(
                np.flatnonzero(nan_mask),
                np.flatnonzero(~nan_mask),
                dem[~nan_mask]
            )
        else:
            # All NaN - return zeros
            return np.zeros_like(dem)
    else:
        dem_filled = dem

    device = _get_device()

    # Convert to tensor: (H, W) -> (1, 1, H, W)
    tensor = torch.from_numpy(dem_filled.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)

    # Horn's method kernels (same as scipy implementation)
    # dx kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] / 8.0
    # dy kernel: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] / 8.0
    dx_kernel = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device
    ).unsqueeze(0).unsqueeze(0) / 8.0

    dy_kernel = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device
    ).unsqueeze(0).unsqueeze(0) / 8.0

    # Apply convolutions with same padding (replicate boundary)
    # Use padding='same' equivalent: pad manually then convolve
    padded = F.pad(tensor, (1, 1, 1, 1), mode='replicate')

    dx = F.conv2d(padded, dx_kernel)
    dy = F.conv2d(padded, dy_kernel)

    # Calculate slope magnitude
    slope = torch.sqrt(dx ** 2 + dy ** 2)

    # Convert back to numpy
    result = slope.squeeze().cpu().numpy()

    # Restore NaN values
    if has_nan:
        result[nan_mask] = np.nan

    return result


def gpu_gaussian_blur(data: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur using GPU acceleration.

    Uses separable 1D convolutions for efficiency. Produces results
    very similar to scipy.ndimage.gaussian_filter.

    Args:
        data: 2D input array (H, W). Can contain NaN values.
        sigma: Standard deviation of Gaussian kernel.

    Returns:
        Blurred array (same shape as input). NaN handling: edges of NaN
        regions may have partial values, centers remain NaN.
    """
    # Handle NaN values
    nan_mask = np.isnan(data)
    has_nan = np.any(nan_mask)

    if has_nan:
        # Fill NaN with local mean for computation
        data_filled = data.copy()
        if np.any(~nan_mask):
            data_filled[nan_mask] = np.nanmean(data)
        else:
            return data.copy()
    else:
        data_filled = data

    device = _get_device()

    # Convert to tensor: (H, W) -> (1, 1, H, W)
    tensor = torch.from_numpy(data_filled.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)

    # Create Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd

    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize

    # Reshape for separable convolution
    kernel_h = kernel_1d.view(1, 1, -1, 1)  # (1, 1, K, 1) for vertical
    kernel_w = kernel_1d.view(1, 1, 1, -1)  # (1, 1, 1, K) for horizontal

    # Apply separable convolution (more efficient than 2D kernel)
    pad_size = kernel_size // 2

    # Vertical pass
    padded = F.pad(tensor, (0, 0, pad_size, pad_size), mode='replicate')
    result = F.conv2d(padded, kernel_h)

    # Horizontal pass
    padded = F.pad(result, (pad_size, pad_size, 0, 0), mode='replicate')
    result = F.conv2d(padded, kernel_w)

    # Convert back to numpy
    output = result.squeeze().cpu().numpy()

    # Restore NaN at center of NaN regions (edges get blurred values)
    if has_nan:
        # Erode the NaN mask to only mark center of NaN regions
        from scipy.ndimage import binary_erosion
        erode_size = max(1, int(sigma))
        if erode_size > 0:
            center_nan = binary_erosion(nan_mask, iterations=erode_size)
            output[center_nan] = np.nan

    return output
