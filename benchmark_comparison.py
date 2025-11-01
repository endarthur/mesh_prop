"""
Benchmark to compare optimized vs unoptimized performance.
Run after implementing vectorization optimizations.
"""

import time
import numpy as np
from mesh_prop import Mesh, points_in_mesh, block_proportions


def create_test_mesh():
    """Create icosahedron mesh for testing."""
    phi = (1 + np.sqrt(5)) / 2
    
    vertices = [
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ]
    
    vertices = np.array(vertices)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    triangles = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    return Mesh(vertices, triangles)


print("=" * 70)
print("Vectorization Optimization Benchmark")
print("=" * 70)
print("\nComparing optimized vectorized implementation")
print("Mesh: Icosahedron (12 vertices, 20 triangles)\n")

mesh = create_test_mesh()

# Test 1: Point-in-mesh with varying point counts
print("-" * 70)
print("Test 1: Point-in-Mesh Performance")
print("-" * 70)

for n_points in [100, 1000, 5000]:
    points = np.random.uniform(-2, 2, (n_points, 3))
    
    start = time.time()
    results = points_in_mesh(mesh, points)
    elapsed = time.time() - start
    
    rate = n_points / elapsed
    inside = np.sum(results)
    
    print(f"{n_points:,} points: {elapsed:.4f}s ({rate:,.0f} points/sec, {inside} inside)")

# Test 2: Block proportions
print("\n" + "-" * 70)
print("Test 2: Block Proportions Performance")
print("-" * 70)

for n_blocks in [10, 100, 500]:
    centroids = np.random.uniform(-1, 1, (n_blocks, 3))
    dims = np.full((n_blocks, 3), 0.3)
    blocks = np.column_stack([centroids, dims])
    
    start = time.time()
    proportions = block_proportions(mesh, blocks, resolution=5)
    elapsed = time.time() - start
    
    rate = n_blocks / elapsed
    print(f"{n_blocks:,} blocks (res=5): {elapsed:.4f}s ({rate:,.0f} blocks/sec)")

print("\n" + "=" * 70)
print("Optimization Summary:")
print("- Vectorized Möller-Trumbore ray-triangle intersection")
print("- Reduced Python loops in favor of NumPy operations")
print("- Early exit optimizations for invalid triangles")
print("=" * 70)
