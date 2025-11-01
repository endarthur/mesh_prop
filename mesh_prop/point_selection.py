"""
Point selection algorithms for determining which points are inside or below a mesh.
"""

import numpy as np


# Numerical tolerance for ray-triangle intersection tests
# This value determines when two floating-point numbers are considered equal
EPSILON = 1e-10

# Ray direction perturbation for numerical stability
# Using irrational-looking numbers helps avoid hitting mesh edges/vertices exactly,
# which would cause ambiguous intersection counts in the ray-casting algorithm
RAY_PERTURBATION_Y = 0.123456789
RAY_PERTURBATION_Z = 0.987654321


def points_in_mesh(mesh, points):
    """
    Determine which points are inside a closed mesh using ray casting algorithm.
    
    This function uses the ray casting algorithm: cast a ray from each point in
    a fixed direction and count how many times it intersects the mesh. If the
    count is odd, the point is inside; if even, it's outside.
    
    Automatically uses BVH acceleration for meshes with more than 100 triangles.
    
    Parameters
    ----------
    mesh : Mesh
        A closed triangular mesh.
    points : array_like, shape (n_points, 3)
        Array of (x, y, z) coordinates to test.
    
    Returns
    -------
    ndarray, shape (n_points,), dtype=bool
        Boolean array where True indicates the point is inside the mesh.
    
    Examples
    --------
    >>> vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    >>> mesh = Mesh(vertices, triangles)
    >>> points = [[0.25, 0.25, 0.25], [2, 2, 2]]
    >>> inside = points_in_mesh(mesh, points)
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    if points.shape[1] != 3:
        raise ValueError(f"points must have shape (n_points, 3), got {points.shape}")
    
    n_points = len(points)
    inside = np.zeros(n_points, dtype=bool)
    
    # Get all triangle vertices at once for vectorization
    tri_verts = mesh.get_all_triangle_vertices()  # shape: (n_triangles, 3, 3)
    
    # Use BVH acceleration if available
    if mesh.bvh is not None:
        for i, point in enumerate(points):
            intersection_count = _count_ray_intersections_bvh(point, mesh.bvh)
            inside[i] = (intersection_count % 2) == 1
    else:
        # Cast a ray in the +x direction from each point
        # For each point, count intersections with triangles
        for i, point in enumerate(points):
            intersection_count = _count_ray_intersections(point, tri_verts)
            inside[i] = (intersection_count % 2) == 1
    
    return inside


def points_below_mesh(mesh, points):
    """
    Determine which points are below an open mesh surface.
    
    A point is considered "below" if it has a smaller z-coordinate than
    the mesh surface directly above it (when projecting down the z-axis).
    
    Automatically uses BVH acceleration for meshes with more than 100 triangles.
    
    Parameters
    ----------
    mesh : Mesh
        An open triangular mesh representing a surface.
    points : array_like, shape (n_points, 3)
        Array of (x, y, z) coordinates to test.
    
    Returns
    -------
    ndarray, shape (n_points,), dtype=bool
        Boolean array where True indicates the point is below the mesh.
    
    Examples
    --------
    >>> vertices = [[0, 0, 1], [1, 0, 1], [0, 1, 1]]
    >>> triangles = [[0, 1, 2]]
    >>> mesh = Mesh(vertices, triangles)
    >>> points = [[0.25, 0.25, 0.5], [0.25, 0.25, 1.5]]
    >>> below = points_below_mesh(mesh, points)
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    if points.shape[1] != 3:
        raise ValueError(f"points must have shape (n_points, 3), got {points.shape}")
    
    n_points = len(points)
    below = np.zeros(n_points, dtype=bool)
    
    # Get all triangle vertices
    tri_verts = mesh.get_all_triangle_vertices()  # shape: (n_triangles, 3, 3)
    
    # Use BVH acceleration if available
    if mesh.bvh is not None:
        for i, point in enumerate(points):
            min_z_above = _find_closest_surface_above_bvh(point, mesh.bvh)
            if min_z_above is not None:
                below[i] = point[2] < min_z_above
    else:
        # For each point, cast a ray upward (+z direction) and find the closest intersection
        for i, point in enumerate(points):
            min_z_above = _find_closest_surface_above(point, tri_verts)
            if min_z_above is not None:
                below[i] = point[2] < min_z_above
    
    return below


def _count_ray_intersections(point, tri_verts):
    """
    Count intersections of a ray from point in +x direction with triangles.
    
    Uses the Möller-Trumbore algorithm for ray-triangle intersection.
    Vectorized across all triangles for better performance.
    
    Parameters
    ----------
    point : ndarray, shape (3,)
        Origin of the ray.
    tri_verts : ndarray, shape (n_triangles, 3, 3)
        Triangle vertices.
    
    Returns
    -------
    int
        Number of intersections.
    """
    ray_origin = point
    # Use a slightly perturbed ray direction to avoid hitting edges/vertices
    # This helps with numerical stability by avoiding degenerate cases
    # where the ray hits exactly on an edge or vertex
    ray_direction = np.array([1.0, RAY_PERTURBATION_Y, RAY_PERTURBATION_Z])
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    
    # Vectorized Möller-Trumbore algorithm across all triangles
    v0 = tri_verts[:, 0, :]  # shape: (n_triangles, 3)
    v1 = tri_verts[:, 1, :]
    v2 = tri_verts[:, 2, :]
    
    # Compute edges
    edge1 = v1 - v0  # shape: (n_triangles, 3)
    edge2 = v2 - v0
    
    # Begin calculating determinant
    h = np.cross(ray_direction, edge2)  # shape: (n_triangles, 3) - FIXED: was np.cross(edge2, ray_direction)
    det = np.sum(edge1 * h, axis=1)  # shape: (n_triangles,)
    
    # Filter out parallel triangles
    valid = np.abs(det) >= EPSILON
    if not np.any(valid):
        return 0
    
    # Continue only with valid triangles
    inv_det = 1.0 / det[valid]
    s = ray_origin - v0[valid]  # shape: (n_valid, 3)
    u = inv_det * np.sum(s * h[valid], axis=1)  # shape: (n_valid,)
    
    # Filter by u parameter
    valid_u = (u >= 0.0) & (u <= 1.0)
    if not np.any(valid_u):
        return 0
    
    # Continue with u-valid triangles
    inv_det = inv_det[valid_u]
    s = s[valid_u]
    edge1_valid = edge1[valid][valid_u]
    edge2_valid = edge2[valid][valid_u]
    
    q = np.cross(s, edge1_valid)  # shape: (n_u_valid, 3)
    v = inv_det * np.sum(ray_direction * q, axis=1)  # shape: (n_u_valid,)
    u = u[valid_u]
    
    # Filter by v parameter
    valid_v = (v >= 0.0) & (u + v <= 1.0)
    if not np.any(valid_v):
        return 0
    
    # Calculate t for final valid triangles
    inv_det = inv_det[valid_v]
    edge2_valid = edge2_valid[valid_v]
    q = q[valid_v]
    
    t = inv_det * np.sum(edge2_valid * q, axis=1)  # shape: (n_final,)
    
    # Count intersections in +x direction
    count = np.sum(t > EPSILON)
    
    return int(count)


def _find_closest_surface_above(point, tri_verts):
    """
    Find the z-coordinate of the closest mesh surface above a point.
    
    Casts a ray from the point in +z direction and finds the closest intersection.
    Vectorized across all triangles for better performance.
    
    Parameters
    ----------
    point : ndarray, shape (3,)
        The query point.
    tri_verts : ndarray, shape (n_triangles, 3, 3)
        Triangle vertices.
    
    Returns
    -------
    float or None
        Z-coordinate of closest surface above, or None if no surface above.
    """
    ray_origin = point
    ray_direction = np.array([0.0, 0.0, 1.0])  # +z direction
    
    # Vectorized Möller-Trumbore algorithm
    v0 = tri_verts[:, 0, :]
    v1 = tri_verts[:, 1, :]
    v2 = tri_verts[:, 2, :]
    
    # Compute edges
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Begin calculating determinant
    h = np.cross(ray_direction, edge2)  # FIXED: was np.cross(edge2, ray_direction)
    det = np.sum(edge1 * h, axis=1)
    
    # Filter out parallel triangles
    valid = np.abs(det) >= EPSILON
    if not np.any(valid):
        return None
    
    # Continue only with valid triangles
    inv_det = 1.0 / det[valid]
    s = ray_origin - v0[valid]
    u = inv_det * np.sum(s * h[valid], axis=1)
    
    # Filter by u parameter
    valid_u = (u >= 0.0) & (u <= 1.0)
    if not np.any(valid_u):
        return None
    
    # Continue with u-valid triangles
    inv_det = inv_det[valid_u]
    s = s[valid_u]
    edge1_valid = edge1[valid][valid_u]
    edge2_valid = edge2[valid][valid_u]
    
    q = np.cross(s, edge1_valid)
    v = inv_det * np.sum(ray_direction * q, axis=1)
    u = u[valid_u]
    
    # Filter by v parameter
    valid_v = (v >= 0.0) & (u + v <= 1.0)
    if not np.any(valid_v):
        return None
    
    # Calculate t for final valid triangles
    inv_det = inv_det[valid_v]
    edge2_valid = edge2_valid[valid_v]
    q = q[valid_v]
    
    t = inv_det * np.sum(edge2_valid * q, axis=1)
    
    # Find closest intersection in +z direction
    valid_t = t > EPSILON
    if not np.any(valid_t):
        return None
    
    min_t = np.min(t[valid_t])
    return ray_origin[2] + min_t


def _count_ray_intersections_bvh(point, bvh):
    """
    Count ray intersections using BVH acceleration.
    
    Parameters
    ----------
    point : ndarray, shape (3,)
        Origin of the ray.
    bvh : BVH
        BVH acceleration structure.
    
    Returns
    -------
    int
        Number of intersections.
    """
    ray_origin = point
    # Use a slightly perturbed ray direction to avoid hitting edges/vertices
    ray_direction = np.array([1.0, RAY_PERTURBATION_Y, RAY_PERTURBATION_Z])
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    
    # Collect all potentially intersecting triangle indices
    candidate_indices = []
    
    def callback(triangle_indices):
        candidate_indices.extend(triangle_indices)
    
    bvh.traverse_ray(ray_origin, ray_direction, callback)
    
    if not candidate_indices:
        return 0
    
    # Test actual intersections with collected triangles
    tri_verts = bvh.triangle_vertices[candidate_indices]
    return _count_ray_intersections(point, tri_verts)


def _find_closest_surface_above_bvh(point, bvh):
    """
    Find closest surface above point using BVH acceleration.
    
    Parameters
    ----------
    point : ndarray, shape (3,)
        The query point.
    bvh : BVH
        BVH acceleration structure.
    
    Returns
    -------
    float or None
        Z-coordinate of closest surface above, or None if no surface above.
    """
    ray_origin = point
    ray_direction = np.array([0.0, 0.0, 1.0])  # +z direction
    
    # Collect all potentially intersecting triangle indices
    candidate_indices = []
    
    def callback(triangle_indices):
        candidate_indices.extend(triangle_indices)
    
    bvh.traverse_ray(ray_origin, ray_direction, callback)
    
    if not candidate_indices:
        return None
    
    # Test actual intersections with collected triangles
    tri_verts = bvh.triangle_vertices[candidate_indices]
    return _find_closest_surface_above(point, tri_verts)

