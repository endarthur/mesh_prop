"""
Tests for block proportion calculations.
"""

import pytest
import numpy as np
from mesh_prop import Mesh, block_proportions


def test_block_proportions_simple_inside():
    """Test block proportions for blocks inside a tetrahedron."""
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
    
    # Block entirely inside (near origin)
    blocks = [
        [[0.05, 0.05, 0.05], [0.15, 0.15, 0.15]]
    ]
    
    proportions = block_proportions(mesh, blocks, method='inside', resolution=3)
    
    # Should be close to 1.0 (entirely inside)
    assert proportions[0] > 0.5


def test_block_proportions_simple_outside():
    """Test block proportions for blocks outside a mesh."""
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
    
    # Block entirely outside
    blocks = [
        [[2, 2, 2], [3, 3, 3]]
    ]
    
    proportions = block_proportions(mesh, blocks, method='inside', resolution=3)
    
    # Should be 0.0 (entirely outside)
    assert proportions[0] == 0.0


def test_block_proportions_multiple_blocks():
    """Test block proportions with multiple blocks."""
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
    
    blocks = [
        [[0.05, 0.05, 0.05], [0.15, 0.15, 0.15]],  # inside
        [[2, 2, 2], [3, 3, 3]]                      # outside
    ]
    
    proportions = block_proportions(mesh, blocks, method='inside', resolution=3)
    
    assert proportions[0] > 0.5  # mostly inside
    assert proportions[1] == 0.0  # entirely outside


def test_block_proportions_below_plane():
    """Test block proportions below a horizontal plane."""
    # Horizontal plane at z=1
    vertices = [
        [0, 0, 1],
        [2, 0, 1],
        [0, 2, 1],
        [2, 2, 1]
    ]
    triangles = [
        [0, 1, 2],
        [1, 2, 3]
    ]
    mesh = Mesh(vertices, triangles)
    
    # Block entirely below the plane
    blocks = [
        [[0.5, 0.5, 0], [1.5, 1.5, 0.5]]
    ]
    
    proportions = block_proportions(mesh, blocks, method='below', resolution=3)
    
    # Should be 1.0 (entirely below)
    assert proportions[0] == 1.0


def test_block_proportions_above_plane():
    """Test block proportions above a horizontal plane."""
    # Horizontal plane at z=1
    vertices = [
        [0, 0, 1],
        [2, 0, 1],
        [0, 2, 1],
        [2, 2, 1]
    ]
    triangles = [
        [0, 1, 2],
        [1, 2, 3]
    ]
    mesh = Mesh(vertices, triangles)
    
    # Block entirely above the plane
    blocks = [
        [[0.5, 0.5, 1.5], [1.5, 1.5, 2]]
    ]
    
    proportions = block_proportions(mesh, blocks, method='below', resolution=3)
    
    # Should be 0.0 (entirely above)
    assert proportions[0] == 0.0


def test_block_proportions_partial():
    """Test block proportions for partially inside/outside."""
    # Simple large tetrahedron
    vertices = [
        [0, 0, 0],
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2]
    ]
    triangles = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ]
    mesh = Mesh(vertices, triangles)
    
    # Block that straddles the mesh boundary
    blocks = [
        [[0.4, 0.4, 0.4], [0.8, 0.8, 0.8]]
    ]
    
    proportions = block_proportions(mesh, blocks, method='inside', resolution=5)
    
    # Should be between 0 and 1
    assert 0.0 < proportions[0] < 1.0


def test_block_proportions_resolution():
    """Test that higher resolution gives different results."""
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
    
    blocks = [
        [[0.1, 0.1, 0.1], [0.4, 0.4, 0.4]]
    ]
    
    prop_low = block_proportions(mesh, blocks, method='inside', resolution=2)
    prop_high = block_proportions(mesh, blocks, method='inside', resolution=10)
    
    # Both should indicate the block is mostly inside, but may differ slightly
    assert prop_low[0] > 0
    assert prop_high[0] > 0


def test_block_proportions_tuple_resolution():
    """Test block proportions with tuple resolution (different per axis)."""
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
    
    # Block inside
    blocks = [
        [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]]
    ]
    
    # Test with tuple resolution (high x, low y, medium z)
    proportions = block_proportions(mesh, blocks, method='inside', resolution=(10, 3, 5))
    
    # Should be mostly inside
    assert proportions[0] > 0.5


def test_block_proportions_tuple_vs_uniform_resolution():
    """Test that tuple and uniform resolution produce consistent results."""
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
    
    blocks = [
        [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]]
    ]
    
    # Compare uniform resolution with equivalent tuple
    prop_uniform = block_proportions(mesh, blocks, method='inside', resolution=5)
    prop_tuple = block_proportions(mesh, blocks, method='inside', resolution=(5, 5, 5))
    
    # Should be identical
    np.testing.assert_array_equal(prop_uniform, prop_tuple)


def test_block_proportions_invalid_method():
    """Test that invalid method raises error."""
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    mesh = Mesh(vertices, triangles)
    
    blocks = [[[0, 0, 0], [1, 1, 1]]]
    
    with pytest.raises(ValueError, match="method must be"):
        block_proportions(mesh, blocks, method='invalid')


def test_block_proportions_invalid_resolution():
    """Test that invalid resolution raises error."""
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    mesh = Mesh(vertices, triangles)
    
    blocks = [[[0, 0, 0], [1, 1, 1]]]
    
    # Test invalid single value
    with pytest.raises(ValueError, match="resolution must be at least"):
        block_proportions(mesh, blocks, resolution=0)
    
    # Test invalid tuple length
    with pytest.raises(ValueError, match="resolution tuple must have 3 elements"):
        block_proportions(mesh, blocks, resolution=(5, 5))
    
    # Test invalid values in tuple
    with pytest.raises(ValueError, match="all resolution values must be at least"):
        block_proportions(mesh, blocks, resolution=(5, 0, 3))


def test_block_proportions_invalid_shape():
    """Test that invalid block shape raises error."""
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    mesh = Mesh(vertices, triangles)
    
    invalid_blocks = [[[0, 0, 0]]]  # Missing second corner
    
    with pytest.raises(ValueError, match="blocks must have shape"):
        block_proportions(mesh, invalid_blocks)


def test_block_proportions_cube():
    """Test block proportions with a cube mesh."""
    # Unit cube from (0,0,0) to (1,1,1)
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
    
    # Block inside the cube
    blocks = [
        [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
    ]
    
    proportions = block_proportions(mesh, blocks, method='inside', resolution=5)
    
    # Should be 1.0 (entirely inside)
    assert proportions[0] == 1.0
