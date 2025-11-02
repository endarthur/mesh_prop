"""
Tests for optional acceleration features (parallel processing and Numba JIT).
"""

import pytest
import numpy as np
from mesh_prop import Mesh, block_proportions, check_numba_available, check_joblib_available


class TestParallelProcessing:
    """Tests for parallel processing in block_proportions."""
    
    def test_parallel_vs_sequential_correctness(self):
        """Test that parallel processing gives the same results as sequential."""
        # Create a simple tetrahedron mesh
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        mesh = Mesh(vertices, triangles)
        
        # Create test blocks
        blocks = [
            [0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
            [0.75, 0.25, 0.25, 0.3, 0.3, 0.3],
            [0.25, 0.75, 0.25, 0.3, 0.3, 0.3],
        ]
        
        # Sequential processing
        props_sequential = block_proportions(
            mesh, blocks, method='inside', resolution=5, auto_optimize=False, n_jobs=1
        )
        
        # Parallel processing (will use sequential if joblib not available)
        props_parallel = block_proportions(
            mesh, blocks, method='inside', resolution=5, auto_optimize=False, n_jobs=2
        )
        
        # Results should be identical
        np.testing.assert_array_almost_equal(props_sequential, props_parallel)
    
    def test_parallel_with_auto_optimize(self):
        """Test that n_jobs is respected when auto_optimize is False."""
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        mesh = Mesh(vertices, triangles)
        
        blocks = [[0.25, 0.25, 0.25, 0.5, 0.5, 0.5]]
        
        # With auto_optimize=False, n_jobs should be used
        props = block_proportions(
            mesh, blocks, method='inside', resolution=5, auto_optimize=False, n_jobs=2
        )
        
        assert len(props) == 1
        assert 0.0 <= props[0] <= 1.0
    
    def test_parallel_auto_jobs(self):
        """Test that n_jobs='auto' works correctly."""
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        mesh = Mesh(vertices, triangles)
        
        blocks = [
            [0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
            [0.75, 0.25, 0.25, 0.3, 0.3, 0.3],
        ]
        
        # Test with n_jobs='auto'
        props = block_proportions(
            mesh, blocks, method='inside', resolution=5, auto_optimize=False, n_jobs='auto'
        )
        
        assert len(props) == 2
        assert all(0.0 <= p <= 1.0 for p in props)
    
    def test_parallel_single_block(self):
        """Test that parallel processing works with single block."""
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        mesh = Mesh(vertices, triangles)
        
        blocks = [[0.25, 0.25, 0.25, 0.5, 0.5, 0.5]]
        
        # Should work fine with single block
        props = block_proportions(
            mesh, blocks, method='inside', resolution=5, auto_optimize=False, n_jobs=2
        )
        
        assert len(props) == 1
        assert 0.0 <= props[0] <= 1.0


class TestAcceleratorAvailability:
    """Tests for checking accelerator availability."""
    
    def test_check_numba_available(self):
        """Test that we can check if Numba is available."""
        # Should return a boolean
        result = check_numba_available()
        assert isinstance(result, bool)
    
    def test_check_joblib_available(self):
        """Test that we can check if joblib is available."""
        # Should return a boolean
        result = check_joblib_available()
        assert isinstance(result, bool)
    
    def test_accelerators_work_without_optional_deps(self):
        """Test that code works even without optional dependencies."""
        # This test just ensures no import errors occur
        from mesh_prop.accelerators import parallel_map, numba_jit
        
        # These should be defined even if dependencies aren't installed
        assert parallel_map is not None
        assert numba_jit is not None
        
        # Test fallback parallel_map
        result = parallel_map(lambda x: x * 2, [1, 2, 3], n_jobs=1)
        assert result == [2, 4, 6]


class TestPerformanceWithOptimizations:
    """Tests that verify optimizations work correctly."""
    
    def test_large_block_set_sequential(self):
        """Test processing many blocks sequentially."""
        # Create a mesh
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        mesh = Mesh(vertices, triangles)
        
        # Create many blocks
        blocks = [[i*0.1, j*0.1, k*0.1, 0.05, 0.05, 0.05]
                  for i in range(5) for j in range(5) for k in range(5)]
        
        # Process sequentially
        props = block_proportions(
            mesh, blocks, method='inside', resolution=3, auto_optimize=False, n_jobs=1
        )
        
        assert len(props) == 125
        assert all(0.0 <= p <= 1.0 for p in props)
    
    def test_large_block_set_parallel(self):
        """Test processing many blocks in parallel."""
        # Create a mesh
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        mesh = Mesh(vertices, triangles)
        
        # Create many blocks
        blocks = [[i*0.1, j*0.1, k*0.1, 0.05, 0.05, 0.05]
                  for i in range(5) for j in range(5) for k in range(5)]
        
        # Process in parallel
        props = block_proportions(
            mesh, blocks, method='inside', resolution=3, auto_optimize=False, n_jobs=2
        )
        
        assert len(props) == 125
        assert all(0.0 <= p <= 1.0 for p in props)
