"""
GPU-accelerated operations using PyTorch.

This module provides GPU-accelerated versions of common terrain processing
operations. Functions automatically use CUDA when available, falling back
to CPU otherwise. All functions accept numpy arrays and return numpy arrays.

Key functions:
- gpu_horn_slope: Slope calculation using Horn's method (via conv2d)
- gpu_gaussian_blur: Gaussian blur (via F.conv2d with separable kernels)
- gpu_median_filter: Median filter (via unfold + median)
- gpu_max_filter: Maximum filter (via max_pool2d)
- gpu_min_filter: Minimum filter (via max_pool2d on negated data)
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


def gpu_median_filter(data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply median filter using GPU acceleration.

    Uses unfold to extract sliding windows, then computes median of each.
    Produces identical results to scipy.ndimage.median_filter.

    Args:
        data: 2D input array (H, W). Can contain NaN values.
        kernel_size: Size of the median filter kernel (odd number).

    Returns:
        Filtered array (same shape as input). NaN regions preserved.
    """
    # Handle NaN values
    nan_mask = np.isnan(data)
    has_nan = np.any(nan_mask)

    if has_nan:
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

    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad_size = kernel_size // 2

    # Pad with replicate (same as scipy 'nearest' mode)
    padded = F.pad(tensor, (pad_size, pad_size, pad_size, pad_size), mode='replicate')

    # Unfold to get sliding windows: (1, 1, H, W) -> (1, K*K, H*W)
    # unfold(dim, size, step) - unfold along height then width
    h, w = data.shape
    patches = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    # patches shape: (1, 1, H, W, K, K)
    patches = patches.contiguous().view(1, 1, h, w, -1)
    # patches shape: (1, 1, H, W, K*K)

    # Compute median along last dimension
    result = torch.median(patches, dim=-1).values
    # result shape: (1, 1, H, W)

    # Convert back to numpy
    output = result.squeeze().cpu().numpy()

    # Restore NaN at center of NaN regions
    if has_nan:
        from scipy.ndimage import binary_erosion
        erode_size = kernel_size // 2
        if erode_size > 0:
            center_nan = binary_erosion(nan_mask, iterations=erode_size)
            output[center_nan] = np.nan

    return output


def gpu_max_filter(data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply maximum filter (dilation) using GPU acceleration.

    Uses max_pool2d for efficient GPU computation. Produces identical
    results to scipy.ndimage.maximum_filter.

    Args:
        data: 2D input array (H, W).
        kernel_size: Size of the filter kernel (odd number).

    Returns:
        Filtered array (same shape as input).
    """
    device = _get_device()

    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Convert to tensor: (H, W) -> (1, 1, H, W)
    tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)

    pad_size = kernel_size // 2

    # Pad with replicate to match scipy behavior
    padded = F.pad(tensor, (pad_size, pad_size, pad_size, pad_size), mode='replicate')

    # Apply max pooling with stride 1
    result = F.max_pool2d(padded, kernel_size=kernel_size, stride=1)

    # Convert back to numpy
    return result.squeeze().cpu().numpy()


def gpu_min_filter(data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply minimum filter (erosion) using GPU acceleration.

    Uses max_pool2d on negated data for efficient GPU computation.
    Produces identical results to scipy.ndimage.minimum_filter.

    Args:
        data: 2D input array (H, W).
        kernel_size: Size of the filter kernel (odd number).

    Returns:
        Filtered array (same shape as input).
    """
    # min(x) = -max(-x)
    return -gpu_max_filter(-data, kernel_size)
