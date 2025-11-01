"""
Block proportion calculations for determining what portion of blocks are inside/below a mesh.
"""

import numpy as np
from .point_selection import points_in_mesh, points_below_mesh


def block_proportions(mesh, blocks, method='inside', resolution=5):
    """
    Calculate what proportion of each block is inside or below a mesh.
    
    This function divides each block into a grid of sample points and tests
    each point to determine the proportion of the block that satisfies the
    condition (inside or below the mesh).
    
    Parameters
    ----------
    mesh : Mesh
        The triangular mesh.
    blocks : array_like, shape (n_blocks, 2, 3)
        Array defining blocks as pairs of opposite corners.
        blocks[i, 0] is the minimum (x, y, z) corner of block i.
        blocks[i, 1] is the maximum (x, y, z) corner of block i.
    method : str, optional
        Either 'inside' (for closed meshes) or 'below' (for open meshes).
        Default is 'inside'.
    resolution : int, optional
        Number of sample points per dimension within each block.
        Higher values give more accurate proportions but are slower.
        Default is 5 (125 points per block).
    
    Returns
    -------
    ndarray, shape (n_blocks,), dtype=float
        Proportion of each block inside/below the mesh, ranging from 0.0 to 1.0.
    
    Examples
    --------
    >>> vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    >>> mesh = Mesh(vertices, triangles)
    >>> blocks = [[[0, 0, 0], [0.5, 0.5, 0.5]], [[0.5, 0.5, 0.5], [1, 1, 1]]]
    >>> proportions = block_proportions(mesh, blocks, method='inside', resolution=3)
    """
    blocks = np.asarray(blocks, dtype=np.float64)
    
    if blocks.ndim != 3 or blocks.shape[1:] != (2, 3):
        raise ValueError(
            f"blocks must have shape (n_blocks, 2, 3), got {blocks.shape}"
        )
    
    if method not in ('inside', 'below'):
        raise ValueError(f"method must be 'inside' or 'below', got {method}")
    
    if resolution < 1:
        raise ValueError(f"resolution must be at least 1, got {resolution}")
    
    n_blocks = len(blocks)
    proportions = np.zeros(n_blocks, dtype=np.float64)
    
    # Generate sample points for each block
    for i, block in enumerate(blocks):
        min_corner, max_corner = block
        
        # Create a grid of sample points
        sample_points = _generate_block_samples(min_corner, max_corner, resolution)
        
        # Test which points satisfy the condition
        if method == 'inside':
            satisfied = points_in_mesh(mesh, sample_points)
        else:  # method == 'below'
            satisfied = points_below_mesh(mesh, sample_points)
        
        # Calculate proportion
        proportions[i] = np.mean(satisfied)
    
    return proportions


def _generate_block_samples(min_corner, max_corner, resolution):
    """
    Generate a regular grid of sample points within a block.
    
    Parameters
    ----------
    min_corner : ndarray, shape (3,)
        Minimum (x, y, z) corner of the block.
    max_corner : ndarray, shape (3,)
        Maximum (x, y, z) corner of the block.
    resolution : int
        Number of sample points per dimension.
    
    Returns
    -------
    ndarray, shape (resolution**3, 3)
        Grid of sample points.
    """
    # Create 1D arrays for each dimension
    x = np.linspace(min_corner[0], max_corner[0], resolution)
    y = np.linspace(min_corner[1], max_corner[1], resolution)
    z = np.linspace(min_corner[2], max_corner[2], resolution)
    
    # Create 3D grid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten and combine
    sample_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    return sample_points
