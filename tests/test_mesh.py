"""
Tests for the Mesh class.
"""

import pytest
import numpy as np
from mesh_prop import Mesh


def test_mesh_creation():
    """Test basic mesh creation."""
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    triangles = [[0, 1, 2]]
    
    mesh = Mesh(vertices, triangles)
    
    assert mesh.n_vertices == 3
    assert mesh.n_triangles == 1
    assert mesh.vertices.shape == (3, 3)
    assert mesh.triangles.shape == (1, 3)


def test_mesh_with_numpy_arrays():
    """Test mesh creation with numpy arrays."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    
    mesh = Mesh(vertices, triangles)
    
    assert mesh.vertices.dtype == np.float64
    assert mesh.triangles.dtype == np.int32


def test_mesh_invalid_vertices_shape():
    """Test that invalid vertex shape raises error."""
    vertices = [[0, 0], [1, 0], [0, 1]]  # Only 2D
    triangles = [[0, 1, 2]]
    
    with pytest.raises(ValueError, match="vertices must have shape"):
        Mesh(vertices, triangles)


def test_mesh_invalid_triangles_shape():
    """Test that invalid triangle shape raises error."""
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    triangles = [[0, 1]]  # Only 2 vertices
    
    with pytest.raises(ValueError, match="triangles must have shape"):
        Mesh(vertices, triangles)


def test_mesh_invalid_triangle_indices():
    """Test that out-of-range triangle indices raise error."""
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    triangles = [[0, 1, 5]]  # Index 5 is out of range
    
    with pytest.raises(ValueError, match="triangle indices must be in range"):
        Mesh(vertices, triangles)


def test_get_triangle_vertices():
    """Test getting vertices of a specific triangle."""
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    triangles = [[0, 1, 2]]
    mesh = Mesh(vertices, triangles)
    
    tri_verts = mesh.get_triangle_vertices(0)
    
    assert tri_verts.shape == (3, 3)
    np.testing.assert_array_equal(tri_verts[0], [0, 0, 0])
    np.testing.assert_array_equal(tri_verts[1], [1, 0, 0])
    np.testing.assert_array_equal(tri_verts[2], [0, 1, 0])


def test_get_all_triangle_vertices():
    """Test getting vertices for all triangles."""
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    triangles = [[0, 1, 2], [0, 1, 3]]
    mesh = Mesh(vertices, triangles)
    
    all_tri_verts = mesh.get_all_triangle_vertices()
    
    assert all_tri_verts.shape == (2, 3, 3)


def test_compute_bounds():
    """Test computing mesh bounding box."""
    vertices = [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]]
    triangles = [[0, 1, 2], [0, 1, 3]]
    mesh = Mesh(vertices, triangles)
    
    min_bounds, max_bounds = mesh.compute_bounds()
    
    np.testing.assert_array_equal(min_bounds, [0, 0, 0])
    np.testing.assert_array_equal(max_bounds, [1, 2, 3])


def test_mesh_repr():
    """Test mesh string representation."""
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    triangles = [[0, 1, 2]]
    mesh = Mesh(vertices, triangles)
    
    repr_str = repr(mesh)
    
    assert "n_vertices=3" in repr_str
    assert "n_triangles=1" in repr_str


def test_tetrahedron_mesh():
    """Test creating a tetrahedron mesh."""
    # Regular tetrahedron
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3)/2, 0],
        [0.5, np.sqrt(3)/6, np.sqrt(2/3)]
    ]
    triangles = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ]
    
    mesh = Mesh(vertices, triangles)
    
    assert mesh.n_vertices == 4
    assert mesh.n_triangles == 4
