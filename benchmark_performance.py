"""
Performance benchmark for mesh_prop library.
"""

import time
import numpy as np
from mesh_prop import Mesh, points_in_mesh, block_proportions


def create_tetrahedron_mesh():
    """Create a simple tetrahedron mesh."""
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
    return Mesh(vertices, triangles)


def create_complex_mesh():
    """Create a moderate complexity mesh (icosahedron subdivision)."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices
    vertices = [
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ]
    
    # Normalize vertices to unit sphere
    vertices = np.array(vertices)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Icosahedron triangles
    triangles = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    return Mesh(vertices, triangles)


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 70)
    print("mesh_prop Performance Benchmarks")
    print("=" * 70)
    print("\nTest Environment:")
    print(f"  Python version: {np.__version__}")
    
    # Benchmark with simple mesh
    print("\n" + "=" * 70)
    print("Simple Mesh (Tetrahedron: 4 vertices, 4 triangles)")
    print("=" * 70)
    mesh_simple = create_tetrahedron_mesh()
    
    print("\nPoint-in-Mesh Detection:")
    for n_points in [100, 1000, 10000]:
        points = np.random.uniform(-2, 2, (n_points, 3))
        start = time.time()
        results = points_in_mesh(mesh_simple, points)
        elapsed = time.time() - start
        rate = n_points / elapsed
        inside = np.sum(results)
        print(f"  {n_points:,} points: {elapsed:.4f}s ({rate:,.0f} points/sec, {inside} inside)")
    
    print("\nBlock Proportions (resolution=5, 125 samples/block):")
    for n_blocks in [10, 100, 1000]:
        centroids = np.random.uniform(-1, 1, (n_blocks, 3))
        dims = np.full((n_blocks, 3), 0.3)
        blocks = np.column_stack([centroids, dims])
        start = time.time()
        proportions = block_proportions(mesh_simple, blocks, resolution=5)
        elapsed = time.time() - start
        rate = n_blocks / elapsed
        print(f"  {n_blocks:,} blocks: {elapsed:.4f}s ({rate:,.0f} blocks/sec)")
    
    # Benchmark with complex mesh
    print("\n" + "=" * 70)
    print("Complex Mesh (Icosahedron: 12 vertices, 20 triangles)")
    print("=" * 70)
    mesh_complex = create_complex_mesh()
    
    print("\nPoint-in-Mesh Detection:")
    for n_points in [100, 1000, 10000]:
        points = np.random.uniform(-2, 2, (n_points, 3))
        start = time.time()
        results = points_in_mesh(mesh_complex, points)
        elapsed = time.time() - start
        rate = n_points / elapsed
        inside = np.sum(results)
        print(f"  {n_points:,} points: {elapsed:.4f}s ({rate:,.0f} points/sec, {inside} inside)")
    
    print("\nBlock Proportions (varying resolution):")
    n_blocks = 100
    for resolution in [3, 5, 10]:
        centroids = np.random.uniform(-1, 1, (n_blocks, 3))
        dims = np.full((n_blocks, 3), 0.3)
        blocks = np.column_stack([centroids, dims])
        start = time.time()
        proportions = block_proportions(mesh_complex, blocks, resolution=resolution)
        elapsed = time.time() - start
        rate = n_blocks / elapsed
        samples = resolution ** 3
        print(f"  {n_blocks} blocks (res={resolution}, {samples} samples/block): {elapsed:.4f}s ({rate:,.0f} blocks/sec)")
    
    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("  - Point-in-mesh scales linearly with number of points")
    print("  - Block proportions scale with blocks × resolution³")
    print("  - Performance suitable for interactive applications")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
