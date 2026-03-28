#pragma once

#include <Eigen/Core>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include <vector>

// Forward declaration -- full definition in material.hpp.
class ActiveComposite;

/// All inputs needed to run inverse design on a single mesh piece.
///
/// Geometry fields (V, F, P, mesh, geometry, MrInv, fixedIdx) typically come
/// from a ParameterizeResult or from a per-patch extraction step.  The caller
/// must ensure that the pointed-to mesh/geometry/ac objects outlive this struct.
struct InverseDesignProblem
{
    // -- Geometry (from ParameterizeResult) -----------------------------------
    Eigen::MatrixXd V;           ///< Target vertices, scaled to platewidth
    Eigen::MatrixXi F;           ///< Triangle faces
    Eigen::MatrixXd P;           ///< 2D parameterization (nV x 2), same scale as V
    geometrycentral::surface::ManifoldSurfaceMesh*    mesh     = nullptr;
    geometrycentral::surface::VertexPositionGeometry* geometry = nullptr;
    geometrycentral::surface::FaceData<Eigen::Matrix2d> MrInv; ///< Per-face inverse rest-shape
    std::vector<int> fixedIdx;   ///< 9 DOF indices for rigid-body removal

    // -- Material model -------------------------------------------------------
    const ActiveComposite* ac = nullptr;

    // -- Solver settings ------------------------------------------------------
    int    max_iter           = 20;
    double epsilon            = 1e-6;
    double w_s                = 1.0;
    double w_b                = 1.0;
    double wM_kap             = 0.1;
    double wL_kap             = 0.1;
    double wM_lam             = 0.0;
    double wL_lam             = 0.1;
    double wP_kap             = 0.01;
    double wP_lam             = 0.01;
    double penalty_threshold  = 0.01;
    double betaP              = 50.0;
    int    patch_id           = -1;  ///< For logging (-1 = whole mesh)
};

/// All outputs from a single inverse design run.
struct InverseDesignResult
{
    Eigen::MatrixXd V_inv;       ///< Continuous inverse design shape
    Eigen::MatrixXd V_proj;      ///< Projected (discrete material) forward shape
    Eigen::VectorXd t1;          ///< Per-face material dose, layer 1
    Eigen::VectorXd t2;          ///< Per-face material dose, layer 2
    Eigen::VectorXd lam_excess;  ///< Per-face lambda feasibility excess
    Eigen::VectorXd kap_excess;  ///< Per-face kappa feasibility excess
    double dist_inv  = 0.0;      ///< MSE of continuous result vs target
    double dist_proj = 0.0;      ///< MSE of projected result vs target
};

/// Run the full SGN inverse design pipeline on a single mesh piece.
///
/// Pipeline:
///   1. ComputeMorphophing -> target (lambda, kappa)
///   2. SGN loop: OptKap / OptLam alternation with joint material penalty
///   3. Material projection to feasible (t1, t2)
///   4. Forward verification from flat initial state
///
/// @param problem  Fully populated problem description.
/// @return         Inverse design result containing shapes, doses, and metrics.
InverseDesignResult runInverseDesign(const InverseDesignProblem& problem);
