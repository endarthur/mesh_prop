"""
Tests for point selection algorithms.
"""

import pytest
import numpy as np
from mesh_prop import Mesh, points_in_mesh, points_below_mesh


def test_points_in_simple_tetrahedron():
    """Test point-in-mesh for a simple tetrahedron."""
    # Simple tetrahedron with vertices at origin and unit axes
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    triangles = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ]
    mesh = Mesh(vertices, triangles)
    
    # Point inside
    inside_point = [[0.25, 0.25, 0.25]]
    result = points_in_mesh(mesh, inside_point)
    assert result[0] == True
    
    # Point outside
    outside_point = [[2, 2, 2]]
    result = points_in_mesh(mesh, outside_point)
    assert result[0] == False


def test_points_in_mesh_multiple_points():
    """Test point-in-mesh with multiple points."""
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    triangles = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ]
    mesh = Mesh(vertices, triangles)
    
    points = [
        [0.25, 0.25, 0.25],  # inside
        [2, 2, 2],            # outside
        [0.1, 0.1, 0.1],      # inside
        [-1, 0, 0]            # outside
    ]
    
    results = points_in_mesh(mesh, points)
    
    assert results[0] == True
    assert results[1] == False
    assert results[2] == True
    assert results[3] == False


def test_points_in_mesh_single_point():
    """Test point-in-mesh with a single point as 1D array."""
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    triangles = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ]
    mesh = Mesh(vertices, triangles)
    
    point = [0.25, 0.25, 0.25]
    result = points_in_mesh(mesh, point)
    
    assert len(result) == 1
    assert result[0] == True


def test_points_in_mesh_on_boundary():
    """Test points on the mesh boundary."""
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    triangles = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ]
    mesh = Mesh(vertices, triangles)
    
    # Point on a face
    point = [[0.5, 0.5, 0]]
    result = points_in_mesh(mesh, point)
    # Boundary points may be inside or outside depending on numerical precision
    assert isinstance(result[0], (bool, np.bool_))


def test_points_below_simple_plane():
    """Test points below a simple horizontal plane."""
    # Horizontal plane at z=1
    vertices = [
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1]
    ]
    triangles = [[0, 1, 2]]
    mesh = Mesh(vertices, triangles)
    
    # Point below the plane
    below_point = [[0.25, 0.25, 0.5]]
    result = points_below_mesh(mesh, below_point)
    assert result[0] == True
    
    # Point above the plane
    above_point = [[0.25, 0.25, 1.5]]
    result = points_below_mesh(mesh, above_point)
    assert result[0] == False


def test_points_below_mesh_multiple_points():
    """Test points below mesh with multiple points."""
    # Horizontal plane at z=1
    vertices = [
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ]
    triangles = [
        [0, 1, 2],
        [1, 2, 3]
    ]
    mesh = Mesh(vertices, triangles)
    
    points = [
        [0.5, 0.5, 0.5],   # below
        [0.5, 0.5, 1.5],   # above
        [0.25, 0.25, 0],   # below
        [2, 2, 0]          # outside projection, should be False
    ]
    
    results = points_below_mesh(mesh, points)
    
    assert results[0] == True
    assert results[1] == False
    assert results[2] == True
    assert results[3] == False


def test_points_below_slanted_surface():
    """Test points below a slanted surface."""
    # Slanted plane
    vertices = [
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 1]
    ]
    triangles = [[0, 1, 2]]
    mesh = Mesh(vertices, triangles)
    
    # Point at (0.25, 0.25, z) - surface should be at z ~ 0.5
    # So z=0.25 should be below
    below_point = [[0.25, 0.25, 0.25]]
    result = points_below_mesh(mesh, below_point)
    assert result[0] == True
    
    # And z=0.75 should be above
    above_point = [[0.25, 0.25, 0.75]]
    result = points_below_mesh(mesh, above_point)
    assert result[0] == False


def test_points_in_mesh_invalid_shape():
    """Test that invalid point shape raises error."""
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    mesh = Mesh(vertices, triangles)
    
    invalid_points = [[0, 0]]  # Only 2D
    
    with pytest.raises(ValueError, match="points must have shape"):
        points_in_mesh(mesh, invalid_points)


def test_points_below_mesh_invalid_shape():
    """Test that invalid point shape raises error."""
    vertices = [[0, 0, 1], [1, 0, 1], [0, 1, 1]]
    triangles = [[0, 1, 2]]
    mesh = Mesh(vertices, triangles)
    
    invalid_points = [[0, 0]]  # Only 2D
    
    with pytest.raises(ValueError, match="points must have shape"):
        points_below_mesh(mesh, invalid_points)


def test_points_in_mesh_cube():
    """Test point-in-mesh for a cube."""
    # Unit cube from (0,0,0) to (1,1,1)
    # Vertices ordered consistently
    vertices = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top
    ]
    # Triangles with consistent outward-facing normals
    triangles = [
        # bottom face (normal pointing down -z)
        [0, 2, 1], [0, 3, 2],
        # top face (normal pointing up +z)
        [4, 5, 6], [4, 6, 7],
        # front face (normal pointing forward -y)
        [0, 1, 5], [0, 5, 4],
        # back face (normal pointing back +y)
        [3, 7, 6], [3, 6, 2],
        # left face (normal pointing left -x)
        [0, 4, 7], [0, 7, 3],
        # right face (normal pointing right +x)
        [1, 2, 6], [1, 6, 5]
    ]
    mesh = Mesh(vertices, triangles)
    
    # Center of cube
    center = [[0.5, 0.5, 0.5]]
    result = points_in_mesh(mesh, center)
    assert result[0] == True
    
    # Outside
    outside = [[1.5, 0.5, 0.5]]
    result = points_in_mesh(mesh, outside)
    assert result[0] == False
