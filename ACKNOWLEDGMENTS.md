# Acknowledgments

## Algorithms and Techniques

This library implements several well-established algorithms and techniques from computer graphics and computational geometry. We acknowledge the original authors and publications:

### Möller-Trumbore Ray-Triangle Intersection Algorithm

**Implementation**: `mesh_prop/point_selection.py`

**Original Publication**:
- **Authors**: Tomas Möller and Ben Trumbore
- **Title**: "Fast, Minimum Storage Ray-Triangle Intersection"
- **Journal**: Journal of Graphics Tools, Vol. 2, No. 1, 1997, pp. 21-28
- **DOI**: 10.1080/10867651.1997.10487468

**Description**: This algorithm provides an efficient method for computing the intersection between a ray and a triangle in 3D space. It is widely used in ray tracing and collision detection applications.

### Bounding Volume Hierarchy (BVH)

**Implementation**: `mesh_prop/bvh.py`

**Description**: BVH is a tree structure on a set of geometric objects. Our implementation uses axis-aligned bounding boxes (AABBs) and is based on standard computer graphics practices. The specific implementation is original to this project.

**References**:
- Kay, T. L., & Kajiya, J. T. (1986). "Ray tracing complex scenes". ACM SIGGRAPH Computer Graphics, 20(4), 269-278.
- Wald, I. (2007). "On fast Construction of SAH-based Bounding Volume Hierarchies". IEEE Symposium on Interactive Ray Tracing.

### Ray Casting for Point-in-Polygon/Mesh

**Implementation**: `mesh_prop/point_selection.py`

**Description**: The ray casting algorithm for point-in-mesh determination is a standard technique in computational geometry. A ray is cast from the query point, and the number of intersections with the mesh determines if the point is inside (odd number of intersections) or outside (even number).

**References**:
- Haines, E. (1994). "Point in Polygon Strategies". Graphics Gems IV, pp. 24-46.
- Sunday, D. (2001). "Inclusion of a Point in a Polygon". Geometry Algorithms.

## Development Tools

This project was developed with assistance from:
- **GitHub Copilot**: AI-powered code completion and generation
- **Python Ecosystem**: NumPy, pytest, and other open-source libraries

## Dependencies

This library builds upon several excellent open-source projects:
- **NumPy** (BSD 3-Clause License): Fundamental package for scientific computing
- **Numba** (BSD 2-Clause License, optional): JIT compiler for Python
- **joblib** (BSD 3-Clause License, optional): Parallel computing utilities
- **pytest** (MIT License): Testing framework

All dependencies are compatible with this project's MIT license.

## License Compliance

This project is licensed under the MIT License (see LICENSE file). All dependencies and referenced algorithms are either:
1. In the public domain (published algorithms and mathematical techniques)
2. Licensed under permissive licenses compatible with MIT
3. Original implementations based on standard practices

## Contributing

If you use algorithms or techniques from published literature in contributions to this project, please:
1. Add appropriate citations in code comments
2. Update this ACKNOWLEDGMENTS file
3. Ensure compatibility with the MIT license

## Contact

For questions about licensing, attribution, or code provenance, please contact:
- **Maintainer**: Arthur Endlein
- **Email**: endarthur@gmail.com
- **Repository**: https://github.com/endarthur/mesh_prop
