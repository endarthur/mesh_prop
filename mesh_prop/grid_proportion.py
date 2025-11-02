"""
Grid-based proportion calculations optimized for dense block models.

This module provides highly efficient proportion calculations for regular grids
of blocks, commonly used in resource modeling. Instead of querying each block
independently, it renders the mesh to a 2D height map and calculates proportions
directly from the height map.
"""

import numpy as np
import warnings


def grid_proportions(mesh, origin, dimensions, n_blocks, method='below', axis='z', mask=None):
    """
    Calculate proportions for a dense regular grid of blocks (optimized for resource modeling).
    
    This function is highly optimized for dense, regular block grids. It renders the mesh
    to a 2D height map perpendicular to the specified axis, then efficiently calculates
    proportions for all blocks from this height map. This is much faster than the general
    block_proportions() function for regular grids.
    
    For irregular or sparse block layouts, use block_proportions() instead.
    
    Parameters
    ----------
    mesh : Mesh
        The triangular mesh.
    origin : array_like, shape (3,)
        Origin point (x, y, z) of the grid (minimum corner).
    dimensions : array_like, shape (3,)
        Block dimensions (dx, dy, dz) - all blocks have the same size.
    n_blocks : array_like, shape (3,)
        Number of blocks (nx, ny, nz) along each axis.
    method : str, optional
        Either 'below' (for open surface meshes) or 'inside' (for closed meshes).
        Default is 'below'.
        - 'below': Calculate proportion of block below the mesh surface
        - 'inside': Calculate proportion of block inside the closed mesh
    axis : str, optional
        Axis perpendicular to the 2D grid plane: 'x', 'y', or 'z'.
        Default is 'z' (grid in xy-plane, heights along z-axis).
        This should be chosen based on the mesh orientation for best results.
    mask : array_like, shape (nx, ny, nz), dtype=bool, optional
        Boolean mask indicating which blocks to calculate proportions for.
        If provided, only blocks where mask[i, j, k] is True will be computed,
        and masked-out blocks will have proportion 0.0. This can significantly
        reduce computation time for sparse grids. Default is None (calculate all blocks).
    
    Returns
    -------
    ndarray, shape (nx, ny, nz), dtype=float
        3D array of proportions for each block, ranging from 0.0 to 1.0.
        Indexed as proportions[i, j, k] for block at position (i, j, k).
        Blocks where mask is False (if mask provided) will have proportion 0.0.
    
    Examples
    --------
    >>> # Create a mesh representing a surface
    >>> vertices = [[0, 0, 0.5], [10, 0, 0.5], [0, 10, 0.5], [10, 10, 0.5]]
    >>> triangles = [[0, 1, 2], [1, 3, 2]]
    >>> mesh = Mesh(vertices, triangles)
    >>> 
    >>> # Calculate proportions for a 10x10x5 grid of blocks
    >>> origin = [0, 0, 0]
    >>> dimensions = [1, 1, 1]  # Each block is 1x1x1
    >>> n_blocks = [10, 10, 5]
    >>> proportions = grid_proportions(mesh, origin, dimensions, n_blocks, method='below')
    >>> # proportions shape: (10, 10, 5)
    >>> 
    >>> # With a mask to compute only specific blocks
    >>> mask = np.zeros((10, 10, 5), dtype=bool)
    >>> mask[5:, 5:, :] = True  # Only compute upper-right quadrant
    >>> proportions = grid_proportions(mesh, origin, dimensions, n_blocks, method='below', mask=mask)
    
    Notes
    -----
    Performance characteristics:
    - Grid rendering: O(grid_cells × triangles)
    - Block proportion calculation: O(total_blocks) or O(masked_blocks) with mask
    - Much faster than block_proportions() for dense grids (100-1000× speedup)
    
    For a 100×100×50 grid (500K blocks):
    - grid_proportions(): Renders 100×100 grid once, then O(1) per block
    - block_proportions(): Queries mesh for each of 500K blocks independently
    
    Using a mask can further improve performance by skipping unnecessary blocks.
    """
    # Validate inputs
    origin = np.asarray(origin, dtype=np.float64)
    dimensions = np.asarray(dimensions, dtype=np.float64)
    n_blocks = np.asarray(n_blocks, dtype=np.int32)
    
    if origin.shape != (3,):
        raise ValueError(f"origin must have shape (3,), got {origin.shape}")
    if dimensions.shape != (3,):
        raise ValueError(f"dimensions must have shape (3,), got {dimensions.shape}")
    if n_blocks.shape != (3,):
        raise ValueError(f"n_blocks must have shape (3,), got {n_blocks.shape}")
    if np.any(dimensions <= 0):
        raise ValueError(f"all dimensions must be positive, got {dimensions}")
    if np.any(n_blocks < 1):
        raise ValueError(f"all n_blocks must be at least 1, got {n_blocks}")
    
    if method not in ('below', 'inside'):
        raise ValueError(f"method must be 'below' or 'inside', got {method}")
    
    if axis not in ('x', 'y', 'z'):
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")
    
    # Validate mask if provided
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        expected_shape = tuple(n_blocks)
        if mask.shape != expected_shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match n_blocks {expected_shape}"
            )
    
    # Map axis to index
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[axis]
    
    # Get the two perpendicular axes
    perp_axes = [i for i in range(3) if i != axis_idx]
    
    # Project mask to 2D if provided (to optimize rendering)
    mask_2d = None
    if mask is not None:
        # Project mask onto 2D grid perpendicular to axis
        # A 2D grid point needs computation if any block along that column is masked True
        if axis_idx == 0:  # x-axis
            mask_2d = np.any(mask, axis=0)  # Project along x
        elif axis_idx == 1:  # y-axis
            mask_2d = np.any(mask, axis=1)  # Project along y
        else:  # z-axis
            mask_2d = np.any(mask, axis=2)  # Project along z
    
    # Render mesh to 2D height map
    if method == 'below':
        height_map = _render_surface_height_map(
            mesh, origin, dimensions, n_blocks, axis_idx, mask_2d
        )
        # Calculate proportions based on height map
        proportions = _calculate_proportions_below(
            height_map, origin, dimensions, n_blocks, axis_idx, mask
        )
    else:  # method == 'inside'
        # For closed meshes, render both top and bottom surfaces
        bottom_map, top_map = _render_closed_mesh_height_maps(
            mesh, origin, dimensions, n_blocks, axis_idx, mask_2d
        )
        # Calculate proportions based on both height maps
        proportions = _calculate_proportions_inside(
            bottom_map, top_map, origin, dimensions, n_blocks, axis_idx, mask
        )
    
    return proportions


def _render_surface_height_map(mesh, origin, dimensions, n_blocks, axis_idx, mask_2d=None):
    """
    Render mesh to a 2D height map showing surface height at each grid point.
    
    Uses ray casting perpendicular to the grid plane to find the surface height.
    
    Parameters
    ----------
    mesh : Mesh
        The triangular mesh.
    origin : ndarray, shape (3,)
        Grid origin.
    dimensions : ndarray, shape (3,)
        Block dimensions.
    n_blocks : ndarray, shape (3,)
        Number of blocks along each axis.
    axis_idx : int
        Index of the axis perpendicular to the grid (0=x, 1=y, 2=z).
    mask_2d : ndarray, shape (n_perp1, n_perp2), dtype=bool, optional
        2D mask indicating which grid points need rendering. If None, render all points.
    
    Returns
    -------
    ndarray, shape (n_perp1, n_perp2)
        Height map with surface heights at each grid point.
        NaN indicates no surface found at that point or masked out point.
    """
    # Get perpendicular axis indices
    perp_axes = [i for i in range(3) if i != axis_idx]
    perp1_idx, perp2_idx = perp_axes
    
    n_perp1 = n_blocks[perp1_idx]
    n_perp2 = n_blocks[perp2_idx]
    
    # Create grid of points in the perpendicular plane
    # We'll cast rays from each grid point along the axis direction
    perp1_coords = origin[perp1_idx] + dimensions[perp1_idx] * (np.arange(n_perp1) + 0.5)
    perp2_coords = origin[perp2_idx] + dimensions[perp2_idx] * (np.arange(n_perp2) + 0.5)
    
    perp1_grid, perp2_grid = np.meshgrid(perp1_coords, perp2_coords, indexing='ij')
    
    # Initialize height map with NaN (no surface found)
    height_map = np.full((n_perp1, n_perp2), np.nan, dtype=np.float64)
    
    # Get mesh triangles
    tri_verts = mesh.get_all_triangle_vertices()
    
    # For each grid point, cast a ray along the axis direction to find surface
    for i in range(n_perp1):
        for j in range(n_perp2):
            # Skip if masked out
            if mask_2d is not None and not mask_2d[i, j]:
                continue
            
            # Create ray origin at this grid point
            ray_origin = np.zeros(3)
            ray_origin[perp1_idx] = perp1_grid[i, j]
            ray_origin[perp2_idx] = perp2_grid[i, j]
            # Start ray well before the grid
            ray_origin[axis_idx] = origin[axis_idx] - 1000 * dimensions[axis_idx]
            
            # Ray direction along the axis
            ray_direction = np.zeros(3)
            ray_direction[axis_idx] = 1.0
            
            # Find all intersections and take the one closest to the origin
            height = _find_surface_height(ray_origin, ray_direction, tri_verts, axis_idx)
            
            if height is not None:
                height_map[i, j] = height
    
    return height_map


def _render_closed_mesh_height_maps(mesh, origin, dimensions, n_blocks, axis_idx, mask_2d=None):
    """
    Render closed mesh to 2D height maps showing bottom and top surfaces.
    
    Parameters
    ----------
    mesh : Mesh
        The closed triangular mesh.
    origin : ndarray, shape (3,)
        Grid origin.
    dimensions : ndarray, shape (3,)
        Block dimensions.
    n_blocks : ndarray, shape (3,)
        Number of blocks along each axis.
    axis_idx : int
        Index of the axis perpendicular to the grid (0=x, 1=y, 2=z).
    mask_2d : ndarray, shape (n_perp1, n_perp2), dtype=bool, optional
        2D mask indicating which grid points need rendering. If None, render all points.
    
    Returns
    -------
    bottom_map : ndarray, shape (n_perp1, n_perp2)
        Height of bottom surface at each grid point.
    top_map : ndarray, shape (n_perp1, n_perp2)
        Height of top surface at each grid point.
    """
    # Get perpendicular axis indices
    perp_axes = [i for i in range(3) if i != axis_idx]
    perp1_idx, perp2_idx = perp_axes
    
    n_perp1 = n_blocks[perp1_idx]
    n_perp2 = n_blocks[perp2_idx]
    
    # Create grid of points
    perp1_coords = origin[perp1_idx] + dimensions[perp1_idx] * (np.arange(n_perp1) + 0.5)
    perp2_coords = origin[perp2_idx] + dimensions[perp2_idx] * (np.arange(n_perp2) + 0.5)
    
    perp1_grid, perp2_grid = np.meshgrid(perp1_coords, perp2_coords, indexing='ij')
    
    # Initialize height maps
    bottom_map = np.full((n_perp1, n_perp2), np.nan, dtype=np.float64)
    top_map = np.full((n_perp1, n_perp2), np.nan, dtype=np.float64)
    
    # Get mesh triangles
    tri_verts = mesh.get_all_triangle_vertices()
    
    # For each grid point, find min and max heights (bottom and top of closed mesh)
    for i in range(n_perp1):
        for j in range(n_perp2):
            # Skip if masked out
            if mask_2d is not None and not mask_2d[i, j]:
                continue
            
            # Create ray origin
            ray_origin = np.zeros(3)
            ray_origin[perp1_idx] = perp1_grid[i, j]
            ray_origin[perp2_idx] = perp2_grid[i, j]
            ray_origin[axis_idx] = origin[axis_idx] - 1000 * dimensions[axis_idx]
            
            # Ray direction along the axis
            ray_direction = np.zeros(3)
            ray_direction[axis_idx] = 1.0
            
            # Find all intersections
            heights = _find_all_surface_heights(ray_origin, ray_direction, tri_verts, axis_idx)
            
            if len(heights) >= 2:
                bottom_map[i, j] = np.min(heights)
                top_map[i, j] = np.max(heights)
            elif len(heights) == 1:
                # Only one surface found - use it for both
                bottom_map[i, j] = heights[0]
                top_map[i, j] = heights[0]
    
    return bottom_map, top_map


def _find_surface_height(ray_origin, ray_direction, tri_verts, axis_idx):
    """Find the first surface intersection height along a ray."""
    from .point_selection import EPSILON
    
    min_t = None
    
    for tri in tri_verts:
        v0, v1, v2 = tri
        
        # Möller-Trumbore ray-triangle intersection
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        h = np.cross(ray_direction, edge2)
        det = np.dot(edge1, h)
        
        if abs(det) < EPSILON:
            continue
        
        inv_det = 1.0 / det
        s = ray_origin - v0
        u = inv_det * np.dot(s, h)
        
        if u < 0.0 or u > 1.0:
            continue
        
        q = np.cross(s, edge1)
        v = inv_det * np.dot(ray_direction, q)
        
        if v < 0.0 or u + v > 1.0:
            continue
        
        t = inv_det * np.dot(edge2, q)
        
        if t > EPSILON:
            if min_t is None or t < min_t:
                min_t = t
    
    if min_t is not None:
        return ray_origin[axis_idx] + min_t * ray_direction[axis_idx]
    return None


def _find_all_surface_heights(ray_origin, ray_direction, tri_verts, axis_idx):
    """Find all surface intersection heights along a ray."""
    from .point_selection import EPSILON
    
    heights = []
    
    for tri in tri_verts:
        v0, v1, v2 = tri
        
        # Möller-Trumbore ray-triangle intersection
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        h = np.cross(ray_direction, edge2)
        det = np.dot(edge1, h)
        
        if abs(det) < EPSILON:
            continue
        
        inv_det = 1.0 / det
        s = ray_origin - v0
        u = inv_det * np.dot(s, h)
        
        if u < 0.0 or u > 1.0:
            continue
        
        q = np.cross(s, edge1)
        v = inv_det * np.dot(ray_direction, q)
        
        if v < 0.0 or u + v > 1.0:
            continue
        
        t = inv_det * np.dot(edge2, q)
        
        if t > EPSILON:
            height = ray_origin[axis_idx] + t * ray_direction[axis_idx]
            heights.append(height)
    
    return heights


def _calculate_proportions_below(height_map, origin, dimensions, n_blocks, axis_idx, mask=None):
    """
    Calculate block proportions below a surface from a height map.
    
    Parameters
    ----------
    height_map : ndarray, shape (n_perp1, n_perp2)
        Surface height at each grid point.
    origin : ndarray, shape (3,)
        Grid origin.
    dimensions : ndarray, shape (3,)
        Block dimensions.
    n_blocks : ndarray, shape (3,)
        Number of blocks along each axis.
    axis_idx : int
        Index of the axis perpendicular to the grid.
    mask : ndarray, shape (nx, ny, nz), dtype=bool, optional
        3D mask indicating which blocks to calculate. If None, calculate all blocks.
    
    Returns
    -------
    ndarray, shape (nx, ny, nz)
        Proportion of each block below the surface.
    """
    perp_axes = [i for i in range(3) if i != axis_idx]
    perp1_idx, perp2_idx = perp_axes
    
    # Create output array with correct shape
    shape = [n_blocks[0], n_blocks[1], n_blocks[2]]
    proportions = np.zeros(shape, dtype=np.float64)
    
    # For each block along the axis direction
    for k in range(n_blocks[axis_idx]):
        # Block extent along axis
        block_min = origin[axis_idx] + k * dimensions[axis_idx]
        block_max = block_min + dimensions[axis_idx]
        block_center = (block_min + block_max) / 2.0
        
        # For each grid point in the perpendicular plane
        for i in range(n_blocks[perp1_idx]):
            for j in range(n_blocks[perp2_idx]):
                # Determine 3D indices based on axis orientation
                if axis_idx == 0:  # x-axis
                    idx_3d = (k, i, j)
                elif axis_idx == 1:  # y-axis
                    idx_3d = (i, k, j)
                else:  # z-axis
                    idx_3d = (i, j, k)
                
                # Skip if masked out
                if mask is not None and not mask[idx_3d]:
                    continue
                
                surface_height = height_map[i, j]
                
                # Calculate proportion below surface
                if np.isnan(surface_height):
                    # No surface found - assume all below (proportion = 1.0)
                    prop = 1.0
                elif surface_height <= block_min:
                    # Surface is below block - nothing below
                    prop = 0.0
                elif surface_height >= block_max:
                    # Surface is above block - all below
                    prop = 1.0
                else:
                    # Surface intersects block
                    prop = (surface_height - block_min) / dimensions[axis_idx]
                
                # Store in correct position based on axis orientation
                if axis_idx == 0:  # x-axis
                    proportions[k, i, j] = prop
                elif axis_idx == 1:  # y-axis
                    proportions[i, k, j] = prop
                else:  # z-axis
                    proportions[i, j, k] = prop
    
    return proportions


def _calculate_proportions_inside(bottom_map, top_map, origin, dimensions, n_blocks, axis_idx, mask=None):
    """
    Calculate block proportions inside a closed mesh from bottom and top height maps.
    
    Parameters
    ----------
    bottom_map : ndarray, shape (n_perp1, n_perp2)
        Bottom surface height at each grid point.
    top_map : ndarray, shape (n_perp1, n_perp2)
        Top surface height at each grid point.
    origin : ndarray, shape (3,)
        Grid origin.
    dimensions : ndarray, shape (3,)
        Block dimensions.
    n_blocks : ndarray, shape (3,)
        Number of blocks along each axis.
    axis_idx : int
        Index of the axis perpendicular to the grid.
    mask : ndarray, shape (nx, ny, nz), dtype=bool, optional
        3D mask indicating which blocks to calculate. If None, calculate all blocks.
    
    Returns
    -------
    ndarray, shape (nx, ny, nz)
        Proportion of each block inside the closed mesh.
    """
    perp_axes = [i for i in range(3) if i != axis_idx]
    perp1_idx, perp2_idx = perp_axes
    
    # Create output array
    shape = [n_blocks[0], n_blocks[1], n_blocks[2]]
    proportions = np.zeros(shape, dtype=np.float64)
    
    # For each block along the axis direction
    for k in range(n_blocks[axis_idx]):
        # Block extent along axis
        block_min = origin[axis_idx] + k * dimensions[axis_idx]
        block_max = block_min + dimensions[axis_idx]
        
        # For each grid point in the perpendicular plane
        for i in range(n_blocks[perp1_idx]):
            for j in range(n_blocks[perp2_idx]):
                # Determine 3D indices based on axis orientation
                if axis_idx == 0:  # x-axis
                    idx_3d = (k, i, j)
                elif axis_idx == 1:  # y-axis
                    idx_3d = (i, k, j)
                else:  # z-axis
                    idx_3d = (i, j, k)
                
                # Skip if masked out
                if mask is not None and not mask[idx_3d]:
                    continue
                
                bottom_height = bottom_map[i, j]
                top_height = top_map[i, j]
                
                # Calculate proportion inside mesh
                if np.isnan(bottom_height) or np.isnan(top_height):
                    # No mesh found at this point
                    prop = 0.0
                else:
                    # Calculate overlap between block and mesh
                    overlap_min = max(block_min, bottom_height)
                    overlap_max = min(block_max, top_height)
                    
                    if overlap_max <= overlap_min:
                        # No overlap
                        prop = 0.0
                    else:
                        # Calculate proportion
                        overlap_height = overlap_max - overlap_min
                        prop = overlap_height / dimensions[axis_idx]
                
                # Store in correct position
                if axis_idx == 0:  # x-axis
                    proportions[k, i, j] = prop
                elif axis_idx == 1:  # y-axis
                    proportions[i, k, j] = prop
                else:  # z-axis
                    proportions[i, j, k] = prop
    
    return proportions
