"""
Example demonstrating grid_proportions for efficient dense block model calculations.

This example shows how to use grid_proportions() for resource modeling applications
where you have a regular grid of blocks and need to calculate proportions relative
to a mesh surface.
"""

import numpy as np
import time
from mesh_prop import Mesh, grid_proportions, block_proportions


def create_topographic_surface():
    """Create a simple topographic surface with some variation."""
    # Create a surface that varies in height
    x = np.linspace(0, 100, 20)
    y = np.linspace(0, 100, 20)
    xx, yy = np.meshgrid(x, y)
    
    # Create height variation (gentle hills)
    zz = 50 + 10 * np.sin(xx / 20) + 10 * np.cos(yy / 20)
    
    # Convert to vertices
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Create triangles (Delaunay-style triangulation of grid)
    triangles = []
    for i in range(19):
        for j in range(19):
            # Two triangles per grid cell
            idx = i * 20 + j
            triangles.append([idx, idx + 1, idx + 20])
            triangles.append([idx + 1, idx + 21, idx + 20])
    
    return Mesh(vertices, triangles)


def example_basic_usage():
    """Basic usage of grid_proportions."""
    print("=" * 70)
    print("Example 1: Basic Grid Proportions Usage")
    print("=" * 70)
    
    # Create a simple flat surface at z=50
    vertices = [
        [0, 0, 50],
        [100, 0, 50],
        [0, 100, 50],
        [100, 100, 50]
    ]
    triangles = [[0, 1, 2], [1, 3, 2]]
    mesh = Mesh(vertices, triangles)
    
    # Define a block model grid
    origin = [0, 0, 0]
    dimensions = [10, 10, 10]  # 10×10×10 meter blocks
    n_blocks = [10, 10, 10]     # 10×10×10 grid
    
    print(f"\nGrid: {n_blocks[0]}×{n_blocks[1]}×{n_blocks[2]} blocks")
    print(f"Block size: {dimensions[0]}×{dimensions[1]}×{dimensions[2]} meters")
    print(f"Total blocks: {np.prod(n_blocks):,}")
    
    # Calculate proportions
    start = time.time()
    proportions = grid_proportions(mesh, origin, dimensions, n_blocks, method='below', axis='z')
    elapsed = time.time() - start
    
    print(f"\nCalculation time: {elapsed:.4f}s")
    print(f"Result shape: {proportions.shape}")
    
    # Analyze results
    print(f"\nStatistics:")
    print(f"  Blocks fully below surface: {np.sum(proportions == 1.0)}")
    print(f"  Blocks fully above surface: {np.sum(proportions == 0.0)}")
    print(f"  Blocks partially intersecting: {np.sum((proportions > 0) & (proportions < 1.0))}")
    print(f"  Mean proportion below: {np.mean(proportions):.3f}")
    
    # Show some specific blocks
    print(f"\nSample blocks:")
    print(f"  Block (5, 5, 2) [20-30m depth]: {proportions[5, 5, 2]:.3f}")
    print(f"  Block (5, 5, 4) [40-50m depth]: {proportions[5, 5, 4]:.3f}")
    print(f"  Block (5, 5, 5) [50-60m depth]: {proportions[5, 5, 5]:.3f}")
    print(f"  Block (5, 5, 7) [70-80m depth]: {proportions[5, 5, 7]:.3f}")


def example_performance_comparison():
    """Compare performance of grid_proportions vs block_proportions."""
    print("\n" + "=" * 70)
    print("Example 2: Performance Comparison")
    print("=" * 70)
    
    # Create a simple surface
    vertices = [
        [0, 0, 25],
        [50, 0, 25],
        [0, 50, 25],
        [50, 50, 25]
    ]
    triangles = [[0, 1, 2], [1, 3, 2]]
    mesh = Mesh(vertices, triangles)
    
    # Test with different grid sizes
    for n in [10, 20, 30]:
        origin = [0, 0, 0]
        dimensions = [5, 5, 5]
        n_blocks_tuple = [n, n, 5]
        total = n * n * 5
        
        print(f"\n{n}×{n}×5 grid ({total:,} blocks):")
        
        # Method 1: grid_proportions (optimized)
        start = time.time()
        proportions_grid = grid_proportions(
            mesh, origin, dimensions, n_blocks_tuple, method='below', axis='z'
        )
        time_grid = time.time() - start
        
        print(f"  grid_proportions: {time_grid:.4f}s ({total/time_grid:.0f} blocks/sec)")
        
        # Method 2: block_proportions (general purpose) - only for smaller grids
        if n <= 20:  # Don't test large grids as it would be too slow
            # Create blocks array for block_proportions
            blocks = []
            for i in range(n_blocks_tuple[0]):
                for j in range(n_blocks_tuple[1]):
                    for k in range(n_blocks_tuple[2]):
                        x = origin[0] + (i + 0.5) * dimensions[0]
                        y = origin[1] + (j + 0.5) * dimensions[1]
                        z = origin[2] + (k + 0.5) * dimensions[2]
                        blocks.append([x, y, z, dimensions[0], dimensions[1], dimensions[2]])
            
            blocks = np.array(blocks)
            
            start = time.time()
            proportions_blocks = block_proportions(mesh, blocks, method='below', resolution=3)
            time_blocks = time.time() - start
            
            print(f"  block_proportions: {time_blocks:.4f}s ({total/time_blocks:.0f} blocks/sec)")
            print(f"  Speedup: {time_blocks/time_grid:.1f}×")
            
            # Verify results are similar
            # Reshape grid result to 1D for comparison
            proportions_grid_flat = proportions_grid.ravel()
            max_diff = np.max(np.abs(proportions_grid_flat - proportions_blocks))
            print(f"  Max difference: {max_diff:.4f}")


def example_different_axes():
    """Demonstrate using different axis orientations."""
    print("\n" + "=" * 70)
    print("Example 3: Different Axis Orientations")
    print("=" * 70)
    
    # Create a vertical surface perpendicular to x-axis at x=50
    vertices = [
        [50, 0, 0],
        [50, 100, 0],
        [50, 0, 100],
        [50, 100, 100]
    ]
    triangles = [[0, 2, 1], [1, 2, 3]]
    mesh = Mesh(vertices, triangles)
    
    origin = [0, 0, 0]
    dimensions = [10, 10, 10]
    n_blocks = [10, 5, 5]
    
    print(f"\nVertical surface at x=50")
    print(f"Grid: {n_blocks[0]}×{n_blocks[1]}×{n_blocks[2]} blocks")
    
    # Calculate with axis='x'
    proportions = grid_proportions(mesh, origin, dimensions, n_blocks, method='below', axis='x')
    
    print(f"\nResults with axis='x' (2D grid in yz-plane):")
    print(f"  Blocks before surface (x<50): {np.sum(proportions[:5, :, :] == 1.0)} / {5*5*5}")
    print(f"  Blocks after surface (x>50): {np.sum(proportions[5:, :, :] == 0.0)} / {5*5*5}")


def example_resource_modeling():
    """Example simulating a real resource modeling scenario."""
    print("\n" + "=" * 70)
    print("Example 4: Resource Modeling Scenario")
    print("=" * 70)
    
    # Create a more realistic topographic surface
    mesh = create_topographic_surface()
    
    print("\nCreated topographic surface with gentle hills")
    print(f"Surface mesh: {mesh.n_vertices} vertices, {mesh.n_triangles} triangles")
    
    # Define block model for a mining project
    origin = [0, 0, 0]
    dimensions = [5, 5, 2.5]  # 5×5×2.5 meter blocks (common in mining)
    n_blocks = [20, 20, 30]    # 20×20×30 grid
    
    total_blocks = np.prod(n_blocks)
    print(f"\nBlock model:")
    print(f"  Grid: {n_blocks[0]}×{n_blocks[1]}×{n_blocks[2]} blocks")
    print(f"  Block size: {dimensions[0]}×{dimensions[1]}×{dimensions[2]} meters")
    print(f"  Total blocks: {total_blocks:,}")
    print(f"  Extent: {n_blocks[0]*dimensions[0]}×{n_blocks[1]*dimensions[1]}×{n_blocks[2]*dimensions[2]} meters")
    
    # Calculate proportions below topography
    start = time.time()
    proportions = grid_proportions(mesh, origin, dimensions, n_blocks, method='below', axis='z')
    elapsed = time.time() - start
    
    print(f"\nCalculation completed in {elapsed:.3f}s ({total_blocks/elapsed:.0f} blocks/sec)")
    
    # Analyze the block model
    print(f"\nBlock model analysis:")
    print(f"  Air blocks (0% below): {np.sum(proportions == 0.0):,} ({100*np.sum(proportions == 0.0)/total_blocks:.1f}%)")
    print(f"  Ground blocks (100% below): {np.sum(proportions == 1.0):,} ({100*np.sum(proportions == 1.0)/total_blocks:.1f}%)")
    print(f"  Surface blocks (0-100%): {np.sum((proportions > 0) & (proportions < 1.0)):,}")
    
    # Calculate volume below surface
    block_volume = np.prod(dimensions)
    total_volume_below = np.sum(proportions) * block_volume
    print(f"\nVolume analysis:")
    print(f"  Block volume: {block_volume} m³")
    print(f"  Total volume below surface: {total_volume_below:,.0f} m³")


if __name__ == "__main__":
    example_basic_usage()
    example_performance_comparison()
    example_different_axes()
    example_resource_modeling()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
