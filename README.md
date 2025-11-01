# mesh_prop

A high-performance Python library for calculating selections and proportions of points and blocks given a triangular mesh.

## Features

- **Point-in-Mesh Detection**: Determine which points are inside a closed mesh using efficient ray-casting algorithm
- **Point-Below-Mesh Detection**: Determine which points are below an open mesh surface
- **Block Proportion Calculation**: Calculate what proportion of each block is inside or below a mesh
- **High Performance**: Uses NumPy for vectorized operations and efficient algorithms
- **Easy to Use**: Simple, intuitive API

## Installation

```bash
pip install -e .
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Mesh Creation

```python
import numpy as np
from mesh_prop import Mesh

# Define vertices (x, y, z coordinates)
vertices = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

# Define triangles (indices into vertices array)
triangles = [
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3]
]

# Create mesh
mesh = Mesh(vertices, triangles)
print(mesh)  # Mesh(n_vertices=4, n_triangles=4)
```

### Point-in-Mesh Detection (Closed Meshes)

```python
from mesh_prop import points_in_mesh

# Test if points are inside the mesh
points = [
    [0.25, 0.25, 0.25],  # inside
    [2, 2, 2],            # outside
]

inside = points_in_mesh(mesh, points)
print(inside)  # [True, False]
```

### Point-Below-Mesh Detection (Open Meshes)

```python
from mesh_prop import Mesh, points_below_mesh

# Create a horizontal plane at z=1
vertices = [
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
]
triangles = [
    [0, 1, 2],
    [1, 2, 3]
]
plane = Mesh(vertices, triangles)

# Test which points are below the plane
points = [
    [0.5, 0.5, 0.5],   # below
    [0.5, 0.5, 1.5],   # above
]

below = points_below_mesh(plane, points)
print(below)  # [True, False]
```

### Block Proportion Calculation

```python
from mesh_prop import block_proportions

# Define blocks as pairs of opposite corners [min_corner, max_corner]
blocks = [
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],  # small block near origin
    [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],  # block far from mesh
]

# Calculate proportions inside the mesh
proportions = block_proportions(mesh, blocks, method='inside', resolution=5)
print(proportions)  # [0.8, 0.0] (80% inside, 0% inside)

# For open meshes, use method='below'
proportions_below = block_proportions(plane, blocks, method='below', resolution=5)
```

## API Reference

### Mesh

```python
Mesh(vertices, triangles)
```

Represents a triangular mesh.

**Parameters:**
- `vertices` (array_like): Shape (n_vertices, 3). Array of (x, y, z) coordinates.
- `triangles` (array_like): Shape (n_triangles, 3). Array of vertex indices for each triangle.

**Attributes:**
- `vertices`: NumPy array of vertex coordinates
- `triangles`: NumPy array of triangle vertex indices
- `n_vertices`: Number of vertices
- `n_triangles`: Number of triangles

### points_in_mesh

```python
points_in_mesh(mesh, points)
```

Determine which points are inside a closed mesh using ray-casting algorithm.

**Parameters:**
- `mesh` (Mesh): A closed triangular mesh
- `points` (array_like): Shape (n_points, 3). Points to test.

**Returns:**
- `ndarray`: Shape (n_points,), dtype=bool. True if point is inside.

### points_below_mesh

```python
points_below_mesh(mesh, points)
```

Determine which points are below an open mesh surface.

**Parameters:**
- `mesh` (Mesh): An open triangular mesh representing a surface
- `points` (array_like): Shape (n_points, 3). Points to test.

**Returns:**
- `ndarray`: Shape (n_points,), dtype=bool. True if point is below the surface.

### block_proportions

```python
block_proportions(mesh, blocks, method='inside', resolution=5)
```

Calculate what proportion of each block is inside or below a mesh.

**Parameters:**
- `mesh` (Mesh): The triangular mesh
- `blocks` (array_like): Shape (n_blocks, 2, 3). Each block defined by [min_corner, max_corner]
- `method` (str): Either 'inside' or 'below'. Default: 'inside'
- `resolution` (int): Number of sample points per dimension. Default: 5

**Returns:**
- `ndarray`: Shape (n_blocks,), dtype=float. Proportion in range [0.0, 1.0]

## Algorithm Details

### Point-in-Mesh (Ray Casting)

The library uses the ray-casting algorithm:
1. Cast a ray from each point in a random direction
2. Count intersections with the mesh triangles
3. If the count is odd, the point is inside; if even, it's outside

Ray-triangle intersection uses the Möller-Trumbore algorithm for efficiency.

### Point-Below-Mesh

For each point:
1. Cast a ray upward in the +z direction
2. Find the closest intersection with the mesh
3. If the point's z-coordinate is less than the intersection, it's below

### Block Proportions

For each block:
1. Generate a regular grid of sample points within the block
2. Test each sample point (using either point-in-mesh or point-below-mesh)
3. Calculate the proportion of satisfied points

Higher resolution values give more accurate results but are slower.

## Performance Considerations

- **NumPy Vectorization**: All operations use NumPy for efficient computation
- **Resolution Trade-off**: For block proportions, higher resolution = more accuracy but slower
- **Mesh Complexity**: Performance scales with the number of triangles in the mesh

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=mesh_prop --cov-report=html
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.