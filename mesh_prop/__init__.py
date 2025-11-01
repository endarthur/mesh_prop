"""
mesh_prop - High-performance library for mesh-based point and block selections.

This library provides efficient algorithms for:
- Determining if points are inside closed meshes
- Determining if points are below open meshes
- Calculating proportions of blocks inside/below meshes
"""

from .mesh import Mesh
from .point_selection import points_in_mesh, points_below_mesh
from .block_proportion import block_proportions

__version__ = "0.1.0"
__all__ = [
    "Mesh",
    "points_in_mesh",
    "points_below_mesh",
    "block_proportions",
]
