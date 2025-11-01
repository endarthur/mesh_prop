"""Tests for grid_detection module."""

import numpy as np
import pytest
from mesh_prop import detect_grid_from_blocks


class TestGridDetection:
    """Tests for detect_grid_from_blocks function."""
    
    def test_perfect_grid_6col(self):
        """Test detection of a perfect regular grid with 6-column blocks."""
        # Create a 5x5x3 grid
        blocks = []
        for i in range(5):
            for j in range(5):
                for k in range(3):
                    blocks.append([i * 1.0, j * 1.0, k * 1.0, 1.0, 1.0, 1.0])
        
        result = detect_grid_from_blocks(blocks)
        
        assert result is not None
        assert result['is_grid'] is True
        assert np.allclose(result['origin'], [-0.5, -0.5, -0.5])
        assert np.allclose(result['dimensions'], [1.0, 1.0, 1.0])
        assert np.array_equal(result['n_blocks'], [5, 5, 3])
        assert result['mask'].shape == (5, 5, 3)
        assert result['mask'].sum() == 75  # All blocks present
    
    def test_perfect_grid_3col(self):
        """Test detection with 3-column blocks and dimensions parameter."""
        # Create a 3x3x2 grid
        blocks = []
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    blocks.append([i * 2.0, j * 2.0, k * 2.0])
        
        result = detect_grid_from_blocks(blocks, dimensions=(2.0, 2.0, 2.0))
        
        assert result is not None
        assert result['is_grid'] is True
        assert np.allclose(result['origin'], [-1.0, -1.0, -1.0])
        assert np.allclose(result['dimensions'], [2.0, 2.0, 2.0])
        assert np.array_equal(result['n_blocks'], [3, 3, 2])
        assert result['mask'].sum() == 18
    
    def test_sparse_grid(self):
        """Test detection of a sparse grid (subset of regular grid)."""
        # Create a checkerboard pattern in a 4x4x2 grid
        blocks = []
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    if (i + j) % 2 == 0:  # Only half the blocks
                        blocks.append([i * 1.0, j * 1.0, k * 1.0, 1.0, 1.0, 1.0])
        
        result = detect_grid_from_blocks(blocks)
        
        assert result is not None
        assert result['is_grid'] is True
        assert result['mask'].shape == (4, 4, 2)
        assert result['mask'].sum() == 16  # Half of 4x4x2=32
    
    def test_irregular_spacing(self):
        """Test that irregular spacing is not detected as a grid."""
        # Create blocks with irregular spacing
        blocks = [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [2.5, 0.0, 0.0, 1.0, 1.0, 1.0],  # Irregular spacing
        ]
        
        result = detect_grid_from_blocks(blocks)
        
        assert result is None
    
    def test_varying_dimensions(self):
        """Test that blocks with varying dimensions are not detected as a grid."""
        # Create blocks with different sizes
        blocks = [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 2.0, 2.0, 2.0],  # Different size
        ]
        
        result = detect_grid_from_blocks(blocks)
        
        assert result is None
    
    def test_single_block(self):
        """Test detection with a single block."""
        blocks = [[5.0, 5.0, 5.0, 1.0, 1.0, 1.0]]
        
        result = detect_grid_from_blocks(blocks)
        
        assert result is not None
        assert result['is_grid'] is True
        assert np.array_equal(result['n_blocks'], [1, 1, 1])
        assert result['mask'].sum() == 1
    
    def test_empty_blocks(self):
        """Test with empty blocks array."""
        blocks = []
        
        result = detect_grid_from_blocks(blocks)
        
        assert result is None
    
    def test_1d_grid(self):
        """Test detection of a 1D grid (line of blocks)."""
        blocks = [[i * 1.0, 0.0, 0.0, 1.0, 1.0, 1.0] for i in range(10)]
        
        result = detect_grid_from_blocks(blocks)
        
        assert result is not None
        assert result['is_grid'] is True
        assert np.array_equal(result['n_blocks'], [10, 1, 1])
    
    def test_2d_grid(self):
        """Test detection of a 2D grid (plane of blocks)."""
        blocks = []
        for i in range(5):
            for j in range(5):
                blocks.append([i * 1.0, j * 1.0, 0.0, 1.0, 1.0, 1.0])
        
        result = detect_grid_from_blocks(blocks)
        
        assert result is not None
        assert result['is_grid'] is True
        assert np.array_equal(result['n_blocks'], [5, 5, 1])
        assert result['mask'].sum() == 25
    
    def test_grid_with_offset(self):
        """Test grid detection with offset origin."""
        # Create a grid starting at (10, 20, 30)
        blocks = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    blocks.append([10 + i * 2.0, 20 + j * 2.0, 30 + k * 2.0, 2.0, 2.0, 2.0])
        
        result = detect_grid_from_blocks(blocks)
        
        assert result is not None
        assert result['is_grid'] is True
        assert np.allclose(result['origin'], [9.0, 19.0, 29.0])
        assert np.allclose(result['dimensions'], [2.0, 2.0, 2.0])
        assert np.array_equal(result['n_blocks'], [3, 3, 3])
