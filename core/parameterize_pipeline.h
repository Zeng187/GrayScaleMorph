#pragma once

#include <Eigen/Core>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include <memory>
#include <vector>

/// Holds all outputs of the parameterization pipeline for a single mesh piece.
///
/// mesh and geometry are heap-allocated because ManifoldSurfaceMesh is not
/// copyable.  Access them through the unique_ptrs.
struct ParameterizeResult
{
    Eigen::MatrixXd V;    ///< 3D target vertices (unchanged from input scale)
    Eigen::MatrixXi F;    ///< Faces (original count -- hole-fill faces removed)
    Eigen::MatrixXd P;    ///< 2D parameterization (gauge-shifted to material window)

    std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh>    mesh;
    std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> geometry;

    geometrycentral::surface::FaceData<Eigen::Matrix2d> MrInv;   ///< per-face 2x2 inverse rest-shape
    std::vector<int> fixedIdx;   ///< 9 DOF indices for rigid-body removal (3 vertices x 3 coords)
};

/// Parameterize a single mesh piece (patch or whole mesh).
///
/// Pipeline: fillInHoles -> Tutte embedding -> constrained ASAP (LocalGlobal)
///           -> centerAndRotate -> restore original face count
///           -> gauge shift (P only) -> build geometry-central objects
///           -> precomputeMrInv -> findCenterFaceIndices.
///
/// V is NOT scaled — it stays in the caller's coordinate system.
/// Only P is gauge-shifted so that per-face λ falls within [lambda_min, lambda_max].
///
/// @param V_in         Input 3D vertices (unchanged in output).
/// @param F_in         Input triangle faces.
/// @param lambda_min   Material-window lower bound on stretch ratio.
/// @param lambda_max   Material-window upper bound on stretch ratio.
/// @return             A fully populated ParameterizeResult (moved out).
ParameterizeResult parameterizeMesh(
    const Eigen::MatrixXd& V_in,
    const Eigen::MatrixXi& F_in,
    double lambda_min,
    double lambda_max);
