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
    /// Poisson ratio for the elastic energy. Sourced from
    /// `Resources/setup/global.json` via `Config::setup.poisson_ratio` and may
    /// be overridden per-experiment by `solver.poisson_ratio` in cfg.json.
    double poisson_ratio      = 0.5;
    int    max_iter           = 20;
    double epsilon            = 1e-6;
    int    verify_max_iter    = 100;  ///< Newton budget for projected-material forward verification (in-loop diagnostic + final).
    double verify_epsilon     = 1e-6; ///< Newton tolerance for projected-material forward verification.
    double w_s                = 1.0;
    double w_b                = 1.0;
    double wM_kap             = 0.1;
    double wL_kap             = 0.1;
    double wM_lam             = 0.0;
    double wL_lam             = 0.1;
    double wP_kap             = 0.01;
    double wP_lam             = 0.01;
    double penalty_threshold  = 0.01;
    /// Lorentzian-soft-min well sharpness for the energy-weighted joint
    /// material penalty (replaces the old log-sum-exp `betaP`).  Larger -> sharper
    /// wells around each feasible (lambda, kappa) point.  Default ~30000 produces
    /// wells of width ~ grid_spacing/3 in energy distance for the default material
    /// window (h=1, nu=0.5, lambda spacing ~0.011, kappa spacing ~0.033).
    double well_scale         = 30000.0;
    /// Maximum number of SGN alternating stages (OptKap + OptLam pairs).
    /// Default 5 was the original setting; increase to give the penalty
    /// more growth steps when wP / well_scale alone are not enough to
    /// drive the continuous solution onto the feasible grid.
    int    max_stages         = 5;
    double wP_growth_factor   = 2.0; ///< Per-stage multiplier for wP_kap/wP_lam when penalty exceeds threshold.
    double wM_decay_factor    = 0.5; ///< Per-stage multiplier for wM_kap and wM_lam (1.0 = keep constant).
    double wL_decay_factor    = 0.5; ///< Per-stage multiplier for wL_kap and wL_lam (1.0 = keep constant).
    int    patch_id           = -1;  ///< For logging (-1 = whole mesh)

    // -- Trajectory CSV logging (diagnostic) --------------------------------
    /// When non-empty, runInverseDesign writes two CSV files inside this
    /// directory (created if missing):
    ///     {trajectory_dir}/{trajectory_tag}_inner.csv  -- per Newton step
    ///     {trajectory_dir}/{trajectory_tag}_outer.csv  -- per OptKap/OptLam/Projected stage
    /// The inner CSV captures every component of the SGN objective at each
    /// Newton iteration; the outer CSV records the resolved continuous and
    /// projected distances + max off-grid residuals at stage boundaries.
    std::string trajectory_dir;
    std::string trajectory_tag;

    // -- Oracle override (twin-experiment diagnostic) -----------------------
    /// If both vectors are non-empty (size = nF), runInverseDesign skips
    /// `Morphmesh::ComputeMorphophing` (which derives target metric from the
    /// deformed shape V) and uses these pre-supplied per-face (lambda, kappa)
    /// directly.  This isolates SGN + projection + forward-verification from
    /// the V -> (lambda, kappa) discretisation residual that always exists in
    /// thin-shell theory (equilibrium V never realises the prescribed metric
    /// exactly).  When the oracle equals the metric used by the Forward run
    /// that produced the target, dist_proj should collapse to ~0.
    Eigen::VectorXd oracle_lambda_pf;
    Eigen::VectorXd oracle_kappa_pf;
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
