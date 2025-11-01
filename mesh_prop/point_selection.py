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
    
    count = 0
    
    for tri in tri_verts:
        v0, v1, v2 = tri
        
        # Compute edges
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Begin calculating determinant
        h = np.cross(ray_direction, edge2)
        det = np.dot(edge1, h)
        
        # Ray is parallel to triangle
        if abs(det) < EPSILON:
            continue
        
        inv_det = 1.0 / det
        s = ray_origin - v0
        u = inv_det * np.dot(s, h)
        
        # Intersection point is outside triangle
        if u < 0.0 or u > 1.0:
            continue
        
        q = np.cross(s, edge1)
        v = inv_det * np.dot(ray_direction, q)
        
        # Intersection point is outside triangle
        if v < 0.0 or u + v > 1.0:
            continue
        
        # Calculate t to find intersection point
        t = inv_det * np.dot(edge2, q)
        
        # Ray intersection (t > EPSILON means intersection is in +x direction)
        if t > EPSILON:
            count += 1
    
    return count


def _find_closest_surface_above(point, tri_verts):
    """
    Find the z-coordinate of the closest mesh surface above a point.
    
    Casts a ray from the point in +z direction and finds the closest intersection.
    
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
    
    min_t = float('inf')
    found = False
    
    for tri in tri_verts:
        v0, v1, v2 = tri
        
        # Compute edges
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Begin calculating determinant
        h = np.cross(ray_direction, edge2)
        det = np.dot(edge1, h)
        
        # Ray is parallel to triangle
        if abs(det) < EPSILON:
            continue
        
        inv_det = 1.0 / det
        s = ray_origin - v0
        u = inv_det * np.dot(s, h)
        
        # Intersection point is outside triangle
        if u < 0.0 or u > 1.0:
            continue
        
        q = np.cross(s, edge1)
        v = inv_det * np.dot(ray_direction, q)
        
        # Intersection point is outside triangle
        if v < 0.0 or u + v > 1.0:
            continue
        
        # Calculate t to find intersection point
        t = inv_det * np.dot(edge2, q)
        
        # Ray intersection in +z direction
        if t > EPSILON and t < min_t:
            min_t = t
            found = True
    
    if found:
        return ray_origin[2] + min_t
    return None
