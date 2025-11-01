"""
Tests for BVH acceleration structure.
"""

import numpy as np
import pytest
from mesh_prop import Mesh, points_in_mesh, points_below_mesh


class TestBVHBasic:
    """Test basic BVH functionality."""
    
    def test_bvh_auto_enabled_large_mesh(self):
        """BVH should auto-enable for meshes with > 100 triangles."""
        # Create a large mesh (e.g., 150 triangles)
        n_triangles = 150
        vertices = np.random.rand(n_triangles * 3, 3)
        triangles = np.arange(n_triangles * 3).reshape(n_triangles, 3)
        
        mesh = Mesh(vertices, triangles)
        assert mesh.bvh is not None, "BVH should auto-enable for 150 triangles"
    
    def test_bvh_disabled_small_mesh(self):
        """BVH should not auto-enable for small meshes."""
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        
        mesh = Mesh(vertices, triangles)
        assert mesh.bvh is None, "BVH should not auto-enable for 4 triangles"
    
    def test_bvh_forced_enable(self):
        """BVH can be forced on for small meshes."""
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        
        mesh = Mesh(vertices, triangles, use_bvh=True)
        assert mesh.bvh is not None, "BVH should be forced on"
    
    def test_bvh_forced_disable(self):
        """BVH can be disabled even for large meshes."""
        n_triangles = 150
        vertices = np.random.rand(n_triangles * 3, 3)
        triangles = np.arange(n_triangles * 3).reshape(n_triangles, 3)
        
        mesh = Mesh(vertices, triangles, use_bvh=False)
        assert mesh.bvh is None, "BVH should be disabled"


class TestBVHCorrectness:
    """Test that BVH produces correct results."""
    
    def test_bvh_tetrahedron_inside(self):
        """BVH and non-BVH should give same results for tetrahedron."""
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        
        mesh_with_bvh = Mesh(vertices, triangles, use_bvh=True)
        mesh_without_bvh = Mesh(vertices, triangles, use_bvh=False)
        
        points = [
            [0.25, 0.25, 0.25],  # inside
            [2, 2, 2],            # outside
            [0.1, 0.1, 0.1],      # inside
            [-0.1, 0.5, 0.5],     # outside
        ]
        
        result_with_bvh = points_in_mesh(mesh_with_bvh, points)
        result_without_bvh = points_in_mesh(mesh_without_bvh, points)
        
        np.testing.assert_array_equal(result_with_bvh, result_without_bvh)
    
    def test_bvh_plane_below(self):
        """BVH and non-BVH should give same results for plane."""
        vertices = [[0, 0, 1], [1, 0, 1], [0, 1, 1]]
        triangles = [[0, 1, 2]]
        
        mesh_with_bvh = Mesh(vertices, triangles, use_bvh=True)
        mesh_without_bvh = Mesh(vertices, triangles, use_bvh=False)
        
        points = [
            [0.25, 0.25, 0.5],  # below
            [0.25, 0.25, 1.5],  # above
            [0.1, 0.1, 0.0],    # below
        ]
        
        result_with_bvh = points_below_mesh(mesh_with_bvh, points)
        result_without_bvh = points_below_mesh(mesh_without_bvh, points)
        
        np.testing.assert_array_equal(result_with_bvh, result_without_bvh)
    
    def test_bvh_large_random_mesh(self):
        """BVH should give same results as non-BVH for large random mesh."""
        np.random.seed(42)
        
        # Create a random mesh
        n_triangles = 200
        vertices = np.random.rand(n_triangles * 3, 3) * 10
        triangles = np.arange(n_triangles * 3).reshape(n_triangles, 3)
        
        mesh_with_bvh = Mesh(vertices, triangles, use_bvh=True)
        mesh_without_bvh = Mesh(vertices, triangles, use_bvh=False)
        
        # Test random points
        points = np.random.rand(50, 3) * 10
        
        result_with_bvh = points_in_mesh(mesh_with_bvh, points)
        result_without_bvh = points_in_mesh(mesh_without_bvh, points)
        
        np.testing.assert_array_equal(result_with_bvh, result_without_bvh)


class TestBVHPerformance:
    """Test BVH performance improvements."""
    
    def test_bvh_stats(self):
        """BVH stats should be reasonable."""
        n_triangles = 200
        vertices = np.random.rand(n_triangles * 3, 3)
        triangles = np.arange(n_triangles * 3).reshape(n_triangles, 3)
        
        mesh = Mesh(vertices, triangles, use_bvh=True)
        stats = mesh.bvh.get_stats()
        
        assert stats['num_nodes'] > 0
        assert stats['num_leaf_nodes'] > 0
        assert stats['depth'] > 0
        assert stats['avg_triangles_per_leaf'] > 0
        assert stats['max_triangles_per_leaf'] <= 10  # Default max
        
        # Tree should be reasonably balanced
        # For 200 triangles with max 10 per leaf, we expect at least 20 leaves
        assert stats['num_leaf_nodes'] >= 20
        
        # Depth should be logarithmic in number of triangles
        # For 200 triangles, depth should be around log2(200/10) ≈ 4-5
        assert stats['depth'] < 20  # Should not be too deep
    
    def test_bvh_correctness_large_mesh(self):
        """BVH should give correct results for large meshes."""
        import time
        
        np.random.seed(42)
        
        # Create a large mesh
        n_triangles = 500
        vertices = np.random.rand(n_triangles * 3, 3) * 10
        triangles = np.arange(n_triangles * 3).reshape(n_triangles, 3)
        
        mesh_with_bvh = Mesh(vertices, triangles, use_bvh=True)
        mesh_without_bvh = Mesh(vertices, triangles, use_bvh=False)
        
        # Test many points
        points = np.random.rand(100, 3) * 10
        
        result_with_bvh = points_in_mesh(mesh_with_bvh, points)
        result_without_bvh = points_in_mesh(mesh_without_bvh, points)
        
        # Results should match
        np.testing.assert_array_equal(result_with_bvh, result_without_bvh)


class TestBVHEdgeCases:
    """Test edge cases for BVH."""
    
    def test_bvh_empty_mesh(self):
        """BVH should handle empty mesh gracefully."""
        vertices = np.array([]).reshape(0, 3)
        triangles = np.array([]).reshape(0, 3)
        
        mesh = Mesh(vertices, triangles, use_bvh=True)
        assert mesh.bvh is None  # No BVH for empty mesh
    
    def test_bvh_single_triangle(self):
        """BVH should work with single triangle."""
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        triangles = [[0, 1, 2]]
        
        mesh = Mesh(vertices, triangles, use_bvh=True)
        assert mesh.bvh is not None
        
        points = [[0.2, 0.2, 0.5], [0.2, 0.2, -0.5]]
        result = points_below_mesh(mesh, points)
        
        assert result[0] == False  # above
        assert result[1] == True   # below
    
    def test_bvh_degenerate_triangles(self):
        """BVH should handle degenerate triangles."""
        # All triangles at same point (degenerate)
        vertices = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        triangles = [[0, 1, 2]]
        
        mesh = Mesh(vertices, triangles, use_bvh=True)
        
        # Should not crash
        points = [[0, 0, 0], [1, 1, 1]]
        result = points_in_mesh(mesh, points)
        
        # Results may vary for degenerate mesh, but should not crash
        assert len(result) == 2
