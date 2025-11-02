"""
Bounding Volume Hierarchy (BVH) for accelerating ray-triangle intersection tests.

BVH organizes triangles into a binary tree of axis-aligned bounding boxes (AABBs).
This dramatically speeds up ray-triangle intersection tests by allowing quick
rejection of large groups of triangles.

For large meshes (1000+ triangles), this provides 10-50× speedup.
"""

import numpy as np


class AABB:
    """Axis-Aligned Bounding Box."""
    
    def __init__(self, min_point, max_point):
        """
        Initialize AABB with min and max corners.
        
        Parameters
        ----------
        min_point : ndarray, shape (3,)
            Minimum (x, y, z) coordinates.
        max_point : ndarray, shape (3,)
            Maximum (x, y, z) coordinates.
        """
        self.min = np.asarray(min_point, dtype=np.float64)
        self.max = np.asarray(max_point, dtype=np.float64)
    
    def intersects_ray(self, ray_origin, ray_inv_direction):
        """
        Test if ray intersects this AABB using slab method.
        
        Parameters
        ----------
        ray_origin : ndarray, shape (3,)
            Ray origin point.
        ray_inv_direction : ndarray, shape (3,)
            Inverse of ray direction (1.0 / ray_direction for each component).
        
        Returns
        -------
        bool
            True if ray intersects AABB.
        """
        # Slab method: compute intersection intervals for each axis
        t_min = (self.min - ray_origin) * ray_inv_direction
        t_max = (self.max - ray_origin) * ray_inv_direction
        
        # Handle negative direction components
        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)
        
        # Find overall interval
        t_near = np.max(t1)
        t_far = np.min(t2)
        
        return t_near <= t_far and t_far >= 0


class BVHNode:
    """Node in the BVH tree."""
    
    def __init__(self, aabb, triangle_indices=None, left=None, right=None):
        """
        Initialize BVH node.
        
        Parameters
        ----------
        aabb : AABB
            Bounding box for this node.
        triangle_indices : list of int, optional
            Triangle indices if this is a leaf node.
        left : BVHNode, optional
            Left child node.
        right : BVHNode, optional
            Right child node.
        """
        self.aabb = aabb
        self.triangle_indices = triangle_indices if triangle_indices is not None else []
        self.left = left
        self.right = right
        self.is_leaf = triangle_indices is not None and (left is None and right is None)


class BVH:
    """
    Bounding Volume Hierarchy for fast ray-triangle intersection.
    
    Organizes mesh triangles into a binary tree of AABBs for efficient
    spatial queries. Automatically built from mesh triangles.
    
    Parameters
    ----------
    triangle_vertices : ndarray, shape (n_triangles, 3, 3)
        All triangle vertices from the mesh.
    max_triangles_per_leaf : int, default=10
        Maximum triangles in a leaf node before splitting.
    
    Attributes
    ----------
    root : BVHNode
        Root node of the BVH tree.
    triangle_vertices : ndarray
        Triangle vertices indexed by the BVH.
    """
    
    def __init__(self, triangle_vertices, max_triangles_per_leaf=10):
        """Build BVH from triangle vertices."""
        self.triangle_vertices = triangle_vertices
        self.max_triangles_per_leaf = max_triangles_per_leaf
        
        n_triangles = len(triangle_vertices)
        if n_triangles == 0:
            self.root = None
            return
        
        # Build tree starting with all triangle indices
        all_indices = list(range(n_triangles))
        self.root = self._build_node(all_indices)
    
    def _compute_aabb(self, triangle_indices):
        """Compute AABB for a set of triangles."""
        if not triangle_indices:
            return AABB(np.zeros(3), np.zeros(3))
        
        # Get all vertices for these triangles
        tri_verts = self.triangle_vertices[triangle_indices]  # shape: (n, 3, 3)
        
        # Flatten to get all vertex coordinates
        all_verts = tri_verts.reshape(-1, 3)  # shape: (n*3, 3)
        
        min_point = np.min(all_verts, axis=0)
        max_point = np.max(all_verts, axis=0)
        
        return AABB(min_point, max_point)
    
    def _choose_split_axis(self, triangle_indices):
        """Choose axis to split on (longest axis of centroid bounds)."""
        if not triangle_indices:
            return 0
        
        # Compute centroids of all triangles
        tri_verts = self.triangle_vertices[triangle_indices]
        centroids = np.mean(tri_verts, axis=1)  # shape: (n, 3)
        
        # Find axis with largest extent
        min_centroid = np.min(centroids, axis=0)
        max_centroid = np.max(centroids, axis=0)
        extents = max_centroid - min_centroid
        
        return np.argmax(extents)
    
    def _build_node(self, triangle_indices):
        """
        Recursively build BVH node.
        
        Parameters
        ----------
        triangle_indices : list of int
            Indices of triangles in this node.
        
        Returns
        -------
        BVHNode
            The constructed node.
        """
        aabb = self._compute_aabb(triangle_indices)
        
        # Leaf node if few enough triangles
        if len(triangle_indices) <= self.max_triangles_per_leaf:
            return BVHNode(aabb, triangle_indices=triangle_indices)
        
        # Choose split axis and find median
        axis = self._choose_split_axis(triangle_indices)
        
        # Compute centroids and sort by chosen axis
        tri_verts = self.triangle_vertices[triangle_indices]
        centroids = np.mean(tri_verts, axis=1)  # shape: (n, 3)
        
        # Sort indices by centroid position on chosen axis
        sorted_indices_order = np.argsort(centroids[:, axis])
        sorted_triangle_indices = [triangle_indices[i] for i in sorted_indices_order]
        
        # Split at median
        mid = len(sorted_triangle_indices) // 2
        left_indices = sorted_triangle_indices[:mid]
        right_indices = sorted_triangle_indices[mid:]
        
        # Handle degenerate case (all triangles have same centroid)
        if not left_indices or not right_indices:
            return BVHNode(aabb, triangle_indices=triangle_indices)
        
        # Recursively build children
        left_node = self._build_node(left_indices)
        right_node = self._build_node(right_indices)
        
        return BVHNode(aabb, left=left_node, right=right_node)
    
    def traverse_ray(self, ray_origin, ray_direction, callback):
        """
        Traverse BVH and call callback for potentially intersecting triangles.
        
        Parameters
        ----------
        ray_origin : ndarray, shape (3,)
            Origin of the ray.
        ray_direction : ndarray, shape (3,)
            Direction of the ray (should be normalized).
        callback : callable
            Function to call with triangle indices. Should accept a list of
            triangle indices and return None. Use this to test actual ray-triangle
            intersections.
        """
        if self.root is None:
            return
        
        # Precompute inverse direction for faster AABB tests
        # Handle zero components to avoid division by zero
        eps = 1e-10
        ray_inv_direction = np.zeros(3)
        for i in range(3):
            if abs(ray_direction[i]) > eps:
                ray_inv_direction[i] = 1.0 / ray_direction[i]
            else:
                ray_inv_direction[i] = 1.0 / eps if ray_direction[i] >= 0 else -1.0 / eps
        
        # Recursively traverse tree
        self._traverse_node(self.root, ray_origin, ray_inv_direction, callback)
    
    def _traverse_node(self, node, ray_origin, ray_inv_direction, callback):
        """Recursively traverse node and its children."""
        if node is None:
            return
        
        # Test ray-AABB intersection
        if not node.aabb.intersects_ray(ray_origin, ray_inv_direction):
            return
        
        # Leaf node: process triangles
        if node.is_leaf:
            if node.triangle_indices:
                callback(node.triangle_indices)
            return
        
        # Interior node: traverse children
        self._traverse_node(node.left, ray_origin, ray_inv_direction, callback)
        self._traverse_node(node.right, ray_origin, ray_inv_direction, callback)
    
    def get_stats(self):
        """
        Get statistics about the BVH tree.
        
        Returns
        -------
        dict
            Dictionary with keys: 'depth', 'num_nodes', 'num_leaf_nodes',
            'avg_triangles_per_leaf', 'max_triangles_per_leaf'
        """
        if self.root is None:
            return {
                'depth': 0,
                'num_nodes': 0,
                'num_leaf_nodes': 0,
                'avg_triangles_per_leaf': 0,
                'max_triangles_per_leaf': 0
            }
        
        def traverse_stats(node, depth=0):
            if node is None:
                return {
                    'max_depth': depth,
                    'num_nodes': 0,
                    'num_leaf_nodes': 0,
                    'leaf_triangle_counts': []
                }
            
            if node.is_leaf:
                return {
                    'max_depth': depth,
                    'num_nodes': 1,
                    'num_leaf_nodes': 1,
                    'leaf_triangle_counts': [len(node.triangle_indices)]
                }
            
            left_stats = traverse_stats(node.left, depth + 1)
            right_stats = traverse_stats(node.right, depth + 1)
            
            return {
                'max_depth': max(left_stats['max_depth'], right_stats['max_depth']),
                'num_nodes': 1 + left_stats['num_nodes'] + right_stats['num_nodes'],
                'num_leaf_nodes': left_stats['num_leaf_nodes'] + right_stats['num_leaf_nodes'],
                'leaf_triangle_counts': left_stats['leaf_triangle_counts'] + right_stats['leaf_triangle_counts']
            }
        
        stats = traverse_stats(self.root)
        leaf_counts = stats['leaf_triangle_counts']
        
        return {
            'depth': stats['max_depth'],
            'num_nodes': stats['num_nodes'],
            'num_leaf_nodes': stats['num_leaf_nodes'],
            'avg_triangles_per_leaf': np.mean(leaf_counts) if leaf_counts else 0,
            'max_triangles_per_leaf': max(leaf_counts) if leaf_counts else 0
        }
