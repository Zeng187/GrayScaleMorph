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
