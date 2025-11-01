"""
Block proportion calculations for determining what portion of blocks are inside/below a mesh.
"""

import warnings
import numpy as np
from .point_selection import points_in_mesh, points_below_mesh


def block_proportions(mesh, blocks, method='inside', resolution=5, dimensions=None):
    """
    Calculate what proportion of each block is inside or below a mesh.
    
    This function divides each block into a grid of sample points and tests
    each point to determine the proportion of the block that satisfies the
    condition (inside or below the mesh).
    
    Parameters
    ----------
    mesh : Mesh
        The triangular mesh.
    blocks : array_like, shape (n_blocks, 3) or (n_blocks, 6)
        Array defining blocks by their centroids and dimensions.
        - If shape is (n_blocks, 3): blocks[i] = [x_centroid, y_centroid, z_centroid].
          In this case, `dimensions` parameter must be provided.
        - If shape is (n_blocks, 6): blocks[i] = [x_centroid, y_centroid, z_centroid, dx, dy, dz].
        Each block is a rectangular box centered at (x_centroid, y_centroid, z_centroid)
        with dimensions dx, dy, dz along x, y, z axes respectively.
    method : str, optional
        Either 'inside' (for closed meshes) or 'below' (for open meshes).
        Default is 'inside'.
    resolution : int or tuple of 3 ints, optional
        Number of sample points per dimension within each block.
        If int, uses the same resolution for all three axes (x, y, z).
        If tuple (res_x, res_y, res_z), uses different resolutions per axis.
        Higher values give more accurate proportions but are slower.
        Default is 5 (125 points per block with uniform resolution).
    dimensions : tuple of 3 floats, optional
        Default dimensions (dx, dy, dz) for all blocks when blocks has shape (n_blocks, 3).
        If provided with 6-column blocks, this parameter takes precedence and a warning is emitted.
        Default is None.
    
    Returns
    -------
    ndarray, shape (n_blocks,), dtype=float
        Proportion of each block inside/below the mesh, ranging from 0.0 to 1.0.
    
    Examples
    --------
    >>> vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    >>> mesh = Mesh(vertices, triangles)
    >>> # Blocks with centroids and individual dimensions
    >>> blocks = [[0.25, 0.25, 0.25, 0.5, 0.5, 0.5], [0.75, 0.75, 0.75, 0.5, 0.5, 0.5]]
    >>> proportions = block_proportions(mesh, blocks, method='inside', resolution=5)
    >>> # Blocks with centroids only, using common dimensions
    >>> blocks = [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
    >>> proportions = block_proportions(mesh, blocks, dimensions=(0.5, 0.5, 0.5), resolution=5)
    >>> # Different resolution per axis
    >>> proportions = block_proportions(mesh, blocks, dimensions=(0.5, 0.5, 0.5), resolution=(10, 5, 3))
    """
    blocks = np.asarray(blocks, dtype=np.float64)
    
    if blocks.ndim != 2:
        raise ValueError(
            f"blocks must be a 2D array, got shape {blocks.shape}"
        )
    
    n_blocks, n_cols = blocks.shape
    
    if n_cols not in (3, 6):
        raise ValueError(
            f"blocks must have 3 or 6 columns, got {n_cols} columns"
        )
    
    # Handle dimensions parameter
    if n_cols == 6 and dimensions is not None:
        warnings.warn(
            "Both 6-column blocks (with individual dimensions) and dimensions parameter provided. "
            "Using dimensions parameter and ignoring the last 3 columns of blocks.",
            UserWarning
        )
        use_dimensions = dimensions
        centroids = blocks[:, :3]
    elif n_cols == 6:
        # Use individual dimensions from blocks
        use_dimensions = None
        centroids = blocks[:, :3]
        individual_dims = blocks[:, 3:6]
    elif n_cols == 3:
        # Must have dimensions parameter
        if dimensions is None:
            raise ValueError(
                "dimensions parameter is required when blocks has 3 columns (centroids only)"
            )
        use_dimensions = dimensions
        centroids = blocks
    
    # Validate dimensions parameter if provided
    if use_dimensions is not None:
        if not isinstance(use_dimensions, (list, tuple)) or len(use_dimensions) != 3:
            raise ValueError(
                f"dimensions must be a tuple/list of 3 floats (dx, dy, dz), got {use_dimensions}"
            )
        dx, dy, dz = use_dimensions
        if dx <= 0 or dy <= 0 or dz <= 0:
            raise ValueError(
                f"all dimension values must be positive, got {use_dimensions}"
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
    
    proportions = np.zeros(n_blocks, dtype=np.float64)
    
    # Generate sample points for each block
    for i in range(n_blocks):
        centroid = centroids[i]
        
        # Get dimensions for this block
        if use_dimensions is not None:
            block_dims = use_dimensions
        else:
            block_dims = individual_dims[i]
        
        # Convert centroid and dimensions to min/max corners
        min_corner = centroid - np.array(block_dims) / 2.0
        max_corner = centroid + np.array(block_dims) / 2.0
        
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
