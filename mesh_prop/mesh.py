"""
Core Mesh class for representing triangular meshes.
"""

import numpy as np


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
    
    Examples
    --------
    >>> vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    >>> mesh = Mesh(vertices, triangles)
    """
    
    def __init__(self, vertices, triangles):
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
    
    def __repr__(self):
        return (
            f"Mesh(n_vertices={self.n_vertices}, "
            f"n_triangles={self.n_triangles})"
        )
