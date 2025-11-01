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
        Each block is defined by two 3D points: [min_corner, max_corner] where
        min_corner = [x_min, y_min, z_min] and max_corner = [x_max, y_max, z_max].
    method : str, optional
        Either 'inside' (for closed meshes) or 'below' (for open meshes).
        Default is 'inside'.
    resolution : int or tuple of 3 ints, optional
        Number of sample points per dimension within each block.
        If int, uses the same resolution for all three axes (x, y, z).
        If tuple (res_x, res_y, res_z), uses different resolutions per axis.
        Higher values give more accurate proportions but are slower.
        Default is 5 (125 points per block with uniform resolution).
    
    Returns
    -------
    ndarray, shape (n_blocks,), dtype=float
        Proportion of each block inside/below the mesh, ranging from 0.0 to 1.0.
    
    Examples
    --------
    >>> vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    >>> mesh = Mesh(vertices, triangles)
    >>> # Single block with uniform resolution
    >>> blocks = [[[0, 0, 0], [0.5, 0.5, 0.5]]]
    >>> proportions = block_proportions(mesh, blocks, method='inside', resolution=3)
    >>> # Multiple blocks with different resolution per axis
    >>> blocks = [[[0, 0, 0], [0.5, 0.5, 0.5]], [[0.5, 0.5, 0.5], [1, 1, 1]]]
    >>> proportions = block_proportions(mesh, blocks, resolution=(5, 3, 7))
    """
    blocks = np.asarray(blocks, dtype=np.float64)
    
    if blocks.ndim != 3 or blocks.shape[1:] != (2, 3):
        raise ValueError(
            f"blocks must have shape (n_blocks, 2, 3), got {blocks.shape}"
        )
    
    if method not in ('inside', 'below'):
        raise ValueError(f"method must be 'inside' or 'below', got {method}")
    
    # Parse resolution parameter
    if isinstance(resolution, (list, tuple)):
        if len(resolution) != 3:
            raise ValueError(
                f"resolution tuple must have 3 elements (x, y, z), got {len(resolution)}"
            )
        res_x, res_y, res_z = resolution
        if res_x < 1 or res_y < 1 or res_z < 1:
            raise ValueError(
                f"all resolution values must be at least 1, got {resolution}"
            )
        resolution_tuple = (int(res_x), int(res_y), int(res_z))
    else:
        if resolution < 1:
            raise ValueError(f"resolution must be at least 1, got {resolution}")
        resolution_tuple = (int(resolution), int(resolution), int(resolution))
    
    n_blocks = len(blocks)
    proportions = np.zeros(n_blocks, dtype=np.float64)
    
    # Generate sample points for each block
    for i, block in enumerate(blocks):
        min_corner, max_corner = block
        
        # Create a grid of sample points
        sample_points = _generate_block_samples(min_corner, max_corner, resolution_tuple)
        
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
    resolution : tuple of 3 ints
        Number of sample points per dimension (res_x, res_y, res_z).
    
    Returns
    -------
    ndarray, shape (res_x * res_y * res_z, 3)
        Grid of sample points.
    """
    res_x, res_y, res_z = resolution
    
    # Create 1D arrays for each dimension
    x = np.linspace(min_corner[0], max_corner[0], res_x)
    y = np.linspace(min_corner[1], max_corner[1], res_y)
    z = np.linspace(min_corner[2], max_corner[2], res_z)
    
    # Create 3D grid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten and combine
    sample_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    return sample_points
