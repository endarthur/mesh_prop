"""
Core Mesh class for representing triangular meshes.
"""

import numpy as np
from .bvh import BVH


# Threshold for auto-enabling BVH acceleration
# Meshes with more triangles than this will automatically use BVH
BVH_AUTO_THRESHOLD = 100


class Mesh:
    """
    Represents a triangular mesh defined by vertices and triangles.
    
    Parameters
    ----------
    vertices : array_like, shape (n_vertices, 3)
        Array containing the (x, y, z) coordinates of each vertex.
    triangles : array_like, shape (n_triangles, 3)
        Array containing the indices of vertices for each triangle.
        Each row contains three indices into the vertices array.
    use_bvh : bool or None, default=None
        Whether to use BVH acceleration for ray-triangle intersection tests.
        If None (default), automatically enables BVH for meshes with more
        than 100 triangles. Set to True to force BVH, or False to disable.
    
    Attributes
    ----------
    vertices : ndarray, shape (n_vertices, 3)
        The vertex coordinates.
    triangles : ndarray, shape (n_triangles, 3)
        The triangle vertex indices.
    n_vertices : int
        Number of vertices in the mesh.
    n_triangles : int
        Number of triangles in the mesh.
    bvh : BVH or None
        BVH acceleration structure if enabled, None otherwise.
    
    Examples
    --------
    >>> vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    >>> mesh = Mesh(vertices, triangles)
    """
    
    def __init__(self, vertices, triangles, use_bvh=None):
        """Initialize mesh with vertices and triangles."""
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int32)
        
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape (n_vertices, 3), got {self.vertices.shape}"
            )
        
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(
                f"triangles must have shape (n_triangles, 3), got {self.triangles.shape}"
            )
        
        if np.any(self.triangles < 0) or np.any(self.triangles >= len(self.vertices)):
            raise ValueError(
                "triangle indices must be in range [0, n_vertices)"
            )
        
        self.n_vertices = len(self.vertices)
        self.n_triangles = len(self.triangles)
        
        # Build BVH if requested or auto-enabled
        self.bvh = None
        if use_bvh is None:
            use_bvh = self.n_triangles > BVH_AUTO_THRESHOLD
        
        if use_bvh and self.n_triangles > 0:
            tri_verts = self.get_all_triangle_vertices()
            self.bvh = BVH(tri_verts)
        
        # Internal caches for reuse (Phase 1: Mesh Caching & Reuse)
        self._height_map_cache = {}  # Cache for grid_proportions height maps
        self._grid_detection_cache = {}  # Cache for grid detection results
    
    def _get_height_map_cache_key(self, origin, dimensions, n_blocks, axis, method):
        """
        Generate a hashable cache key for height map caching.
        
        Parameters
        ----------
        origin : array_like
            Grid origin coordinates.
        dimensions : array_like
            Block dimensions.
        n_blocks : array_like
            Number of blocks along each axis.
        axis : str
            Axis perpendicular to the grid ('x', 'y', or 'z').
        method : str
            Method type ('below' or 'inside').
        
        Returns
        -------
        tuple
            Hashable cache key.
        """
        origin_tuple = tuple(np.asarray(origin, dtype=np.float64).flatten())
        dims_tuple = tuple(np.asarray(dimensions, dtype=np.float64).flatten())
        nblocks_tuple = tuple(np.asarray(n_blocks, dtype=np.int32).flatten())
        return (origin_tuple, dims_tuple, nblocks_tuple, axis, method)
    
    def get_cached_height_map(self, origin, dimensions, n_blocks, axis, method):
        """
        Retrieve cached height map if available.
        
        Returns
        -------
        dict or None
            Cached height map data or None if not found.
        """
        cache_key = self._get_height_map_cache_key(origin, dimensions, n_blocks, axis, method)
        return self._height_map_cache.get(cache_key)
    
    def cache_height_map(self, origin, dimensions, n_blocks, axis, method, height_map_data):
        """
        Cache a computed height map for future reuse.
        
        Parameters
        ----------
        origin, dimensions, n_blocks, axis, method : various
            Parameters that define the height map.
        height_map_data : dict
            Dictionary containing height map(s) and metadata.
        """
        cache_key = self._get_height_map_cache_key(origin, dimensions, n_blocks, axis, method)
        self._height_map_cache[cache_key] = height_map_data
    
    def clear_caches(self):
        """Clear all internal caches. Useful if mesh is modified externally."""
        self._height_map_cache.clear()
        self._grid_detection_cache.clear()
    
    def get_triangle_vertices(self, triangle_idx):
        """
        Get the vertices of a specific triangle.
        
        Parameters
        ----------
        triangle_idx : int
            Index of the triangle.
        
        Returns
        -------
        ndarray, shape (3, 3)
            Array where each row is a vertex (x, y, z) of the triangle.
        """
        indices = self.triangles[triangle_idx]
        return self.vertices[indices]
    
    def get_all_triangle_vertices(self):
        """
        Get vertices for all triangles.
        
        Returns
        -------
        ndarray, shape (n_triangles, 3, 3)
            Array where result[i, j] is the j-th vertex of the i-th triangle.
        """
        return self.vertices[self.triangles]
    
    def compute_bounds(self):
        """
        Compute the bounding box of the mesh.
        
        Returns
        -------
        min_bounds : ndarray, shape (3,)
            Minimum (x, y, z) coordinates.
        max_bounds : ndarray, shape (3,)
            Maximum (x, y, z) coordinates.
        """
        min_bounds = np.min(self.vertices, axis=0)
        max_bounds = np.max(self.vertices, axis=0)
        return min_bounds, max_bounds
    
    def clear_cache(self):
        """
        Clear all internal caches.
        
        Call this method if the mesh is modified externally.
        Note: Modifying mesh data directly is not recommended.
        """
        self._height_map_cache.clear()
        self._grid_detection_cache.clear()
    
    def __repr__(self):
        return (
            f"Mesh(n_vertices={self.n_vertices}, "
            f"n_triangles={self.n_triangles})"
        )
