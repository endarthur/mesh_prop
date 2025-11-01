"""
Utility functions for detecting if blocks form a regular grid.

This module provides functions to detect whether an array of blocks with fixed
dimensions actually represents a subset of a regular grid, which allows for
optimization using grid_proportions instead of block_proportions.
"""

import numpy as np


def detect_grid_from_blocks(blocks, dimensions=None, tolerance=1e-6):
    """
    Detect if blocks form a subset of a regular grid and return grid parameters.
    
    This function analyzes an array of blocks to determine if they are positioned
    on a regular grid with uniform spacing. If so, it returns the grid parameters
    that can be used with grid_proportions() for massive performance improvements.
    
    Parameters
    ----------
    blocks : array_like, shape (n_blocks, 3) or (n_blocks, 6)
        Array of blocks with centroids or centroids+dimensions.
        - If shape is (n_blocks, 3): blocks[i] = [x_centroid, y_centroid, z_centroid]
        - If shape is (n_blocks, 6): blocks[i] = [x_centroid, y_centroid, z_centroid, dx, dy, dz]
    dimensions : tuple of 3 floats, optional
        Block dimensions (dx, dy, dz) if blocks only contains centroids.
        Required if blocks has shape (n_blocks, 3).
    tolerance : float, optional
        Tolerance for comparing floating point grid spacing.
        Default is 1e-6.
    
    Returns
    -------
    dict or None
        If blocks form a regular grid subset, returns a dictionary with:
        - 'is_grid': True
        - 'origin': (x, y, z) minimum corner of the bounding grid
        - 'dimensions': (dx, dy, dz) block dimensions
        - 'n_blocks': (nx, ny, nz) number of blocks along each axis
        - 'mask': boolean array of shape (nx, ny, nz) indicating which blocks are present
        
        If blocks do not form a regular grid, returns None.
    
    Examples
    --------
    >>> # Regular grid blocks
    >>> blocks = [[i, j, k] for i in range(10) for j in range(10) for k in range(5)]
    >>> result = detect_grid_from_blocks(blocks, dimensions=(1.0, 1.0, 1.0))
    >>> if result['is_grid']:
    ...     print(f"Grid detected: {result['n_blocks']} blocks")
    
    >>> # Sparse grid (subset of regular grid)
    >>> sparse_blocks = [[i, j, 0] for i in range(10) for j in range(10) if (i+j) % 2 == 0]
    >>> result = detect_grid_from_blocks(sparse_blocks, dimensions=(1.0, 1.0, 1.0))
    >>> if result['is_grid']:
    ...     print(f"Sparse grid: mask has {result['mask'].sum()} active blocks")
    """
    blocks = np.asarray(blocks, dtype=np.float64)
    
    if blocks.ndim != 2:
        return None
    
    n_blocks_total, n_cols = blocks.shape
    
    if n_cols == 6:
        centroids = blocks[:, :3]
        block_dims = blocks[:, 3:6]
        
        # Check if all blocks have the same dimensions
        if not np.allclose(block_dims, block_dims[0], rtol=tolerance, atol=tolerance):
            return None
        
        dimensions = tuple(block_dims[0])
    elif n_cols == 3:
        centroids = blocks
        if dimensions is None:
            return None
        dimensions = tuple(dimensions)
    else:
        return None
    
    if n_blocks_total == 0:
        return None
    
    # Extract x, y, z coordinates
    x_coords = centroids[:, 0]
    y_coords = centroids[:, 1]
    z_coords = centroids[:, 2]
    
    # Find unique coordinates along each axis
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    z_unique = np.unique(z_coords)
    
    # Check if coordinates form a regular grid
    def check_regular_spacing(coords, dim_size):
        """Check if coordinates are regularly spaced with spacing equal to dim_size."""
        if len(coords) < 2:
            return True, coords[0] - dim_size / 2.0 if len(coords) == 1 else 0.0
        
        # Calculate spacing between consecutive unique values
        spacings = np.diff(coords)
        
        # Check if all spacings are equal to the block dimension
        if not np.allclose(spacings, dim_size, rtol=tolerance, atol=tolerance):
            return False, None
        
        # Origin is the minimum coordinate minus half block size
        origin_coord = coords[0] - dim_size / 2.0
        return True, origin_coord
    
    is_x_regular, x_origin = check_regular_spacing(x_unique, dimensions[0])
    is_y_regular, y_origin = check_regular_spacing(y_unique, dimensions[1])
    is_z_regular, z_origin = check_regular_spacing(z_unique, dimensions[2])
    
    if not (is_x_regular and is_y_regular and is_z_regular):
        return None
    
    # Calculate grid parameters
    nx = len(x_unique)
    ny = len(y_unique)
    nz = len(z_unique)
    
    origin = np.array([x_origin, y_origin, z_origin])
    n_blocks_grid = np.array([nx, ny, nz], dtype=np.int32)
    
    # Create mask array
    mask = np.zeros((nx, ny, nz), dtype=bool)
    
    # Map each block centroid to its grid position
    for centroid in centroids:
        # Calculate grid indices
        i = int(np.round((centroid[0] - x_origin - dimensions[0]/2.0) / dimensions[0]))
        j = int(np.round((centroid[1] - y_origin - dimensions[1]/2.0) / dimensions[1]))
        k = int(np.round((centroid[2] - z_origin - dimensions[2]/2.0) / dimensions[2]))
        
        # Validate indices
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            mask[i, j, k] = True
    
    # Check that we found all blocks in the mask
    if mask.sum() != n_blocks_total:
        # Some blocks didn't map correctly, probably not a regular grid
        return None
    
    return {
        'is_grid': True,
        'origin': origin,
        'dimensions': dimensions,
        'n_blocks': n_blocks_grid,
        'mask': mask
    }
