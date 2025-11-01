"""Tests for grid_proportion module."""

import numpy as np
import pytest
from mesh_prop import Mesh, grid_proportions


class TestGridProportions:
    """Tests for grid_proportions function."""
    
    def test_flat_surface_below(self):
        """Test with a flat horizontal surface at z=0.5."""
        # Create a flat surface at z=0.5
        vertices = [
            [0, 0, 0.5],
            [10, 0, 0.5],
            [0, 10, 0.5],
            [10, 10, 0.5]
        ]
        triangles = [[0, 1, 2], [1, 3, 2]]
        mesh = Mesh(vertices, triangles)
        
        # Create a 5x5x5 grid from (0,0,0) to (5,5,5)
        origin = [0, 0, 0]
        dimensions = [1, 1, 1]
        n_blocks = [5, 5, 5]
        
        proportions = grid_proportions(
            mesh, origin, dimensions, n_blocks, method='below', axis='z'
        )
        
        # Check shape
        assert proportions.shape == (5, 5, 5)
        
        # Blocks below z=0.5 should be fully below (proportion=1.0)
        assert np.allclose(proportions[:, :, 0], 1.0)  # z=0-1, all below
        
        # Block at z=0-1 intersects surface at 0.5, so proportion=0.5
        # Actually the block center is at 0.5, so it depends on implementation
        # Let's check block 1 (z=1-2) should be partially below
        
    def test_flat_surface_below_simple(self):
        """Test with simple flat surface."""
        # Surface at z=1.5
        vertices = [
            [-10, -10, 1.5],
            [10, -10, 1.5],
            [-10, 10, 1.5],
            [10, 10, 1.5]
        ]
        triangles = [[0, 1, 2], [1, 3, 2]]
        mesh = Mesh(vertices, triangles)
        
        # 2x2x3 grid
        origin = [0, 0, 0]
        dimensions = [1, 1, 1]
        n_blocks = [2, 2, 3]
        
        proportions = grid_proportions(
            mesh, origin, dimensions, n_blocks, method='below', axis='z'
        )
        
        # Block 0: z=0-1, fully below surface at 1.5
        assert np.allclose(proportions[:, :, 0], 1.0)
        
        # Block 1: z=1-2, surface at 1.5 means 0.5/1.0 = 0.5 below
        assert np.allclose(proportions[:, :, 1], 0.5)
        
        # Block 2: z=2-3, fully above surface
        assert np.allclose(proportions[:, :, 2], 0.0)
    
    def test_closed_mesh_inside(self):
        """Test with a closed box mesh."""
        # Create a simple box from (1,1,1) to (3,3,3)
        vertices = [
            [1, 1, 1], [3, 1, 1], [3, 3, 1], [1, 3, 1],  # bottom
            [1, 1, 3], [3, 1, 3], [3, 3, 3], [1, 3, 3],  # top
        ]
        triangles = [
            # bottom
            [0, 1, 2], [0, 2, 3],
            # top
            [4, 6, 5], [4, 7, 6],
            # sides
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ]
        mesh = Mesh(vertices, triangles)
        
        # Create grid from (0,0,0) with 1x1x1 blocks
        origin = [0, 0, 0]
        dimensions = [1, 1, 1]
        n_blocks = [4, 4, 4]
        
        proportions = grid_proportions(
            mesh, origin, dimensions, n_blocks, method='inside', axis='z'
        )
        
        # Check blocks completely inside the box
        # Box spans from (1,1,1) to (3,3,3)
        # Block at [1,1,1] spans (1,1,1) to (2,2,2), partially inside
        # Block at [2,2,2] spans (2,2,2) to (3,3,3), partially inside
        
        # Blocks outside the box should have proportion 0
        assert proportions[0, 0, 0] == 0.0  # (0,0,0)-(1,1,1)
        assert proportions[3, 3, 3] == 0.0  # (3,3,3)-(4,4,4)
    
    def test_different_axes(self):
        """Test with different axis orientations."""
        # Vertical surface perpendicular to x-axis at x=1.5
        vertices = [
            [1.5, 0, 0],
            [1.5, 10, 0],
            [1.5, 0, 10],
            [1.5, 10, 10]
        ]
        triangles = [[0, 2, 1], [1, 2, 3]]
        mesh = Mesh(vertices, triangles)
        
        # Test with axis='x'
        origin = [0, 0, 0]
        dimensions = [1, 1, 1]
        n_blocks = [3, 2, 2]
        
        proportions = grid_proportions(
            mesh, origin, dimensions, n_blocks, method='below', axis='x'
        )
        
        assert proportions.shape == (3, 2, 2)
        
        # Block at x=0-1 should be fully "below" (before) the surface
        assert np.allclose(proportions[0, :, :], 1.0)
        
        # Block at x=1-2 should be 0.5 below (surface at 1.5)
        assert np.allclose(proportions[1, :, :], 0.5)
        
        # Block at x=2-3 should be fully above (after) the surface
        assert np.allclose(proportions[2, :, :], 0.0)
    
    def test_input_validation(self):
        """Test input validation."""
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        triangles = [[0, 1, 2]]
        mesh = Mesh(vertices, triangles)
        
        # Invalid origin shape
        with pytest.raises(ValueError, match="origin must have shape"):
            grid_proportions(mesh, [0, 0], [1, 1, 1], [2, 2, 2])
        
        # Invalid dimensions shape
        with pytest.raises(ValueError, match="dimensions must have shape"):
            grid_proportions(mesh, [0, 0, 0], [1, 1], [2, 2, 2])
        
        # Invalid n_blocks shape
        with pytest.raises(ValueError, match="n_blocks must have shape"):
            grid_proportions(mesh, [0, 0, 0], [1, 1, 1], [2, 2])
        
        # Negative dimensions
        with pytest.raises(ValueError, match="all dimensions must be positive"):
            grid_proportions(mesh, [0, 0, 0], [1, -1, 1], [2, 2, 2])
        
        # Zero n_blocks
        with pytest.raises(ValueError, match="all n_blocks must be at least 1"):
            grid_proportions(mesh, [0, 0, 0], [1, 1, 1], [2, 0, 2])
        
        # Invalid method
        with pytest.raises(ValueError, match="method must be"):
            grid_proportions(mesh, [0, 0, 0], [1, 1, 1], [2, 2, 2], method='invalid')
        
        # Invalid axis
        with pytest.raises(ValueError, match="axis must be"):
            grid_proportions(mesh, [0, 0, 0], [1, 1, 1], [2, 2, 2], axis='w')
    
    def test_small_grid(self):
        """Test with a small grid to verify basic functionality."""
        # Simple triangle mesh
        vertices = [[0, 0, 0.5], [2, 0, 0.5], [1, 2, 0.5]]
        triangles = [[0, 1, 2]]
        mesh = Mesh(vertices, triangles)
        
        origin = [0, 0, 0]
        dimensions = [1, 1, 1]
        n_blocks = [2, 2, 1]
        
        proportions = grid_proportions(
            mesh, origin, dimensions, n_blocks, method='below', axis='z'
        )
        
        assert proportions.shape == (2, 2, 1)
        # Block at z=0-1 with surface at z=0.5 should be 0.5 below
        # (where the triangle covers)
        assert proportions[0, 0, 0] >= 0.0
        assert proportions[0, 0, 0] <= 1.0
    
    def test_y_axis(self):
        """Test with y-axis perpendicular to grid."""
        # Horizontal surface perpendicular to y at y=1.5
        vertices = [
            [0, 1.5, 0],
            [10, 1.5, 0],
            [0, 1.5, 10],
            [10, 1.5, 10]
        ]
        triangles = [[0, 1, 2], [1, 3, 2]]
        mesh = Mesh(vertices, triangles)
        
        origin = [0, 0, 0]
        dimensions = [1, 1, 1]
        n_blocks = [2, 3, 2]
        
        proportions = grid_proportions(
            mesh, origin, dimensions, n_blocks, method='below', axis='y'
        )
        
        assert proportions.shape == (2, 3, 2)
        
        # y=0-1, fully below
        assert np.allclose(proportions[:, 0, :], 1.0)
        
        # y=1-2, surface at 1.5, so 0.5 below
        assert np.allclose(proportions[:, 1, :], 0.5)
        
        # y=2-3, fully above
        assert np.allclose(proportions[:, 2, :], 0.0)


class TestGridProportionsPerformance:
    """Performance tests for grid_proportions."""
    
    def test_large_grid_performance(self):
        """Test performance with a larger grid (not too large for CI)."""
        import time
        
        # Create a simple surface
        vertices = [
            [-100, -100, 5],
            [100, -100, 5],
            [-100, 100, 5],
            [100, 100, 5]
        ]
        triangles = [[0, 1, 2], [1, 3, 2]]
        mesh = Mesh(vertices, triangles)
        
        # 50x50x10 grid = 25,000 blocks
        origin = [0, 0, 0]
        dimensions = [1, 1, 1]
        n_blocks = [50, 50, 10]
        
        start = time.time()
        proportions = grid_proportions(
            mesh, origin, dimensions, n_blocks, method='below', axis='z'
        )
        elapsed = time.time() - start
        
        assert proportions.shape == (50, 50, 10)
        
        # Should complete in reasonable time (< 5 seconds for 25K blocks)
        assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s for 25K blocks"
        
        print(f"\nPerformance: {len(proportions.ravel())} blocks in {elapsed:.3f}s "
              f"({len(proportions.ravel())/elapsed:.0f} blocks/sec)")
