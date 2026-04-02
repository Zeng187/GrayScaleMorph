#pragma once

#include <Eigen/Core>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include <vector>

/// Identify boundary faces (faces with at least one boundary edge).
std::vector<bool> identifyBoundaryFaces(geometrycentral::surface::SurfaceMesh& mesh);

/// Build reference-face mapping for boundary faces.
///
/// Interior faces (all edges interior) map to themselves.
/// Boundary faces map to the nearest interior face by BFS on the face dual
/// graph (traversal only through non-boundary edges).
///
/// If an unreachable boundary face exists (disconnected from any interior
/// face), it maps to itself and a warning is logged.
std::vector<int> buildRefFaces(geometrycentral::surface::SurfaceMesh& mesh, std::vector<bool>& is_boundary);

/// Precompute the discrete shape operator L (3x3, ambient space) for every
/// face, using ref_faces to redirect boundary faces to interior stencils.
///
/// For interior faces (ref == self): L is computed from the face's own
/// halfedge dihedral angles (identical to the original inline computation).
///
/// For boundary faces: L is computed from ref_faces[fi]'s neighborhood,
/// giving a complete stencil.
///
/// @param mesh      Surface mesh
/// @param V         Current 3D vertex positions (nV x 3)
/// @param F         Face-vertex index matrix (nF x 3)
/// @param ref_faces Mapping from buildRefFaces()
/// @return          Per-face 3x3 shape operator in ambient coordinates
geometrycentral::surface::FaceData<Eigen::Matrix3d>
precomputeShapeOperators(
    geometrycentral::surface::SurfaceMesh& mesh,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& ref_faces);
