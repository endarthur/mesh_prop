"""
Example usage of the mesh_prop library.

This script demonstrates the main features of the library with practical examples.
"""

import numpy as np
from mesh_prop import Mesh, points_in_mesh, points_below_mesh, block_proportions


def example_tetrahedron():
    """Example: Point-in-mesh with a tetrahedron."""
    print("=" * 60)
    print("Example 1: Point-in-Mesh with Tetrahedron")
    print("=" * 60)
    
    # Create a simple tetrahedron
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
    print(f"Created mesh: {mesh}")
    
    # Test various points
    points = [
        [0.25, 0.25, 0.25],  # clearly inside
        [0.1, 0.1, 0.1],      # inside
        [0.5, 0.5, 0.5],      # outside (beyond the slanted face)
        [2, 2, 2],            # clearly outside
        [-1, 0, 0]            # outside
    ]
    
    results = points_in_mesh(mesh, points)
    
    print("\nPoint-in-mesh results:")
    for point, inside in zip(points, results):
        status = "INSIDE" if inside else "OUTSIDE"
        print(f"  Point {point}: {status}")


def example_horizontal_plane():
    """Example: Points below a horizontal plane."""
    print("\n" + "=" * 60)
    print("Example 2: Points Below a Horizontal Plane")
    print("=" * 60)
    
    # Create a horizontal plane at z=1
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
    plane = Mesh(vertices, triangles)
    print(f"Created plane mesh: {plane}")
    
    # Test points at different heights
    points = [
        [1, 1, 0.5],   # below
        [1, 1, 0.0],   # below
        [1, 1, 1.5],   # above
        [1, 1, 2.0],   # above
        [0.5, 0.5, 0.99]  # just below
    ]
    
    results = points_below_mesh(plane, points)
    
    print("\nPoints below plane (z=1):")
    for point, below in zip(points, results):
        status = "BELOW" if below else "ABOVE/OUTSIDE"
        print(f"  Point {point}: {status}")


def example_block_proportions():
    """Example: Block proportions with a tetrahedron."""
    print("\n" + "=" * 60)
    print("Example 3: Block Proportions")
    print("=" * 60)
    
    # Create a tetrahedron
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
    print(f"Created mesh: {mesh}")
    
    # Define several blocks
    blocks = [
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],   # near origin, mostly inside
        [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],   # middle, partially inside
        [[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]],   # far corner, mostly outside
        [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],   # completely outside
    ]
    
    # Calculate proportions with different resolutions
    print("\nBlock proportions (method='inside'):")
    for resolution in [3, 5, 10]:
        proportions = block_proportions(mesh, blocks, method='inside', resolution=resolution)
        print(f"\n  Uniform resolution={resolution}:")
        for i, (block, prop) in enumerate(zip(blocks, proportions)):
            print(f"    Block {i} {block[0]} -> {block[1]}: {prop:.2%}")
    
    # Calculate with tuple resolution (different per axis)
    print("\n  Tuple resolution=(10, 5, 3) [high x, medium y, low z]:")
    proportions = block_proportions(mesh, blocks, method='inside', resolution=(10, 5, 3))
    for i, (block, prop) in enumerate(zip(blocks, proportions)):
        print(f"    Block {i} {block[0]} -> {block[1]}: {prop:.2%}")


def example_cube():
    """Example: A complete cube mesh."""
    print("\n" + "=" * 60)
    print("Example 4: Unit Cube")
    print("=" * 60)
    
    # Unit cube from (0,0,0) to (1,1,1)
    vertices = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top
    ]
    triangles = [
        [0, 2, 1], [0, 3, 2],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [3, 7, 6], [3, 6, 2],  # back
        [0, 4, 7], [0, 7, 3],  # left
        [1, 2, 6], [1, 6, 5]   # right
    ]
    mesh = Mesh(vertices, triangles)
    print(f"Created cube mesh: {mesh}")
    
    # Test points
    points = [
        [0.5, 0.5, 0.5],   # center
        [0.25, 0.25, 0.25],  # inside
        [0.99, 0.99, 0.99],  # near corner, inside
        [1.5, 0.5, 0.5],     # outside
    ]
    
    results = points_in_mesh(mesh, points)
    print("\nPoint-in-cube results:")
    for point, inside in zip(points, results):
        status = "INSIDE" if inside else "OUTSIDE"
        print(f"  Point {point}: {status}")
    
    # Test block proportion
    blocks = [
        [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],  # entirely inside
    ]
    proportions = block_proportions(mesh, blocks, method='inside', resolution=5)
    print(f"\nBlock [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]: {proportions[0]:.2%} inside")


def main():
    """Run all examples."""
    print("\nmesh_prop Library Examples")
    print("=" * 60)
    
    example_tetrahedron()
    example_horizontal_plane()
    example_block_proportions()
    example_cube()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
