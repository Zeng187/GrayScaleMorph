#include "parameterize_pipeline.h"

#include "LocalGlobalSolver.h"
#include "morph_functions.hpp"
#include "parameterization.h"   // fillInHoles, tutteEmbedding, centerAndRotate
#include "simulation_utils.h"   // findCenterFaceIndices

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Anonymous-namespace helper: gauge-shift scale computation
// ---------------------------------------------------------------------------

namespace {

/// Compute the optimal gauge scale t* for ASAP parameterization.
///
/// Solves:  t* = argmin_t  SUM_f  A_f * (t*lam_f - clamp(t*lam_f, lam_min, lam_max))^2
///
/// After calling, the caller should scale P /= t* so that the per-face
/// stretch ratios move into the material window [lambda_min, lambda_max].
double computeGaugeShiftScale(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& P,
    double lambda_min,
    double lambda_max)
{
    const int nF = F.rows();
    Eigen::VectorXd lambda_raw = Eigen::VectorXd::Ones(nF);
    Eigen::VectorXd face_area  = Eigen::VectorXd::Zero(nF);

    for (int fi = 0; fi < nF; ++fi) {
        Eigen::Vector3d v0 = V.row(F(fi, 0));
        Eigen::Vector3d v1 = V.row(F(fi, 1));
        Eigen::Vector3d v2 = V.row(F(fi, 2));
        Eigen::Matrix<double, 3, 2> M;
        M.col(0) = v1 - v0;
        M.col(1) = v2 - v0;

        Eigen::Vector2d p0 = P.row(F(fi, 0));
        Eigen::Vector2d p1 = P.row(F(fi, 1));
        Eigen::Vector2d p2 = P.row(F(fi, 2));
        Eigen::Matrix2d Mr;
        Mr.col(0) = p1 - p0;
        Mr.col(1) = p2 - p0;

        const double det_Mr = Mr.determinant();
        face_area(fi) = 0.5 * std::abs(det_Mr);

        if (std::abs(det_Mr) < 1e-16) {
            lambda_raw(fi) = 1.0;
            continue;
        }

        Eigen::Matrix<double, 3, 2> Fg = M * Mr.inverse();
        Eigen::Matrix2d a = Fg.transpose() * Fg;
        lambda_raw(fi) = std::sqrt(std::max(0.0, 0.5 * a.trace()));
    }

    // Initial guess: align geometric mean to window centre
    double log_sum  = 0.0;
    double area_sum = 0.0;
    for (int fi = 0; fi < nF; ++fi) {
        if (lambda_raw(fi) > 0.0 && face_area(fi) > 0.0) {
            log_sum  += face_area(fi) * std::log(lambda_raw(fi));
            area_sum += face_area(fi);
        }
    }
    if (area_sum <= 0.0) return 1.0;

    const double geom_mean = std::exp(log_sum / area_sum);
    if (!(geom_mean > 0.0) || !std::isfinite(geom_mean)) return 1.0;

    double t = std::sqrt(lambda_min * lambda_max) / geom_mean;
    if (!(t > 0.0) || !std::isfinite(t)) return 1.0;

    // Iterative refinement: solve dF/dt = 0 with set partitioning
    for (int iter = 0; iter < 5; ++iter) {
        double num = 0.0;
        double den = 0.0;
        bool any_outside = false;

        for (int fi = 0; fi < nF; ++fi) {
            const double tl = t * lambda_raw(fi);
            const double A  = face_area(fi);
            const double l  = lambda_raw(fi);

            if (tl < lambda_min) {
                num += A * l * lambda_min;
                den += A * l * l;
                any_outside = true;
            } else if (tl > lambda_max) {
                num += A * l * lambda_max;
                den += A * l * l;
                any_outside = true;
            }
        }

        if (!any_outside || den <= 0.0) break;

        const double t_new = num / den;
        if (!(t_new > 0.0) || !std::isfinite(t_new)) break;
        if (std::abs(t_new - t) < 1e-10 * std::max(1.0, std::abs(t))) {
            t = t_new;
            break;
        }
        t = t_new;
    }

    return t;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

ParameterizeResult parameterizeMesh(
    const Eigen::MatrixXd& V_in,
    const Eigen::MatrixXi& F_in,
    double lambda_min,
    double lambda_max,
    double platewidth)
{
    using namespace geometrycentral::surface;

    // Work on copies -- the inputs are const
    Eigen::MatrixXd V = V_in;
    Eigen::MatrixXi F = F_in;
    const int nF_orig = F.rows();

    // 1. Fill internal holes (keep the longest boundary loop open)
    std::vector<int> boundary = fillInHoles(V, F);

    // 2. Tutte embedding on the filled mesh
    Eigen::MatrixXd P = tutteEmbedding(V, F, boundary);

    // 3. Constrained ASAP via local-global solver
    //    Singular-value bounds are the *inverse* of the material stretch window.
    LocalGlobalSolver solver(V, F);
    solver.solve(P, 1.0 / lambda_max, 1.0 / lambda_min);

    // 4. Centre and align the 2D parameterization with the 3D mesh
    centerAndRotate(V, P);

    // 5. Restore original face count (drop the hole-fill triangles)
    F.conservativeResize(nF_orig, 3);

    // 6. Gauge shift: globally scale P so that the stretch distribution
    //    lands inside the material window [lambda_min, lambda_max].
    const double t = computeGaugeShiftScale(V, F, P, lambda_min, lambda_max);
    if (t > 0.0 && std::isfinite(t) && std::abs(t - 1.0) > 1e-12) {
        P /= t;
        spdlog::info("Gauge shift t={:.6f}, P scaled by {:.6f}", t, 1.0 / t);
    }

    // 7. Platewidth scaling: fit parameterization bbox to platewidth,
    //    apply the same factor to V so that V and P stay in the same scale.
    const double scale =
        platewidth / (P.colwise().maxCoeff() - P.colwise().minCoeff()).maxCoeff();
    V *= scale;
    P *= scale;

    // 8. Build geometry-central mesh and geometry from the scaled V, F
    auto mesh     = std::make_unique<ManifoldSurfaceMesh>(F);
    auto geometry = std::make_unique<VertexPositionGeometry>(*mesh, V);
    geometry->refreshQuantities();

    // 9. Precompute per-face inverse rest-shape from the 2D parameterization
    FaceData<Eigen::Matrix2d> MrInv = precomputeMrInv(*mesh, P, F);

    // 10. Identify centre-face DOF indices for rigid-body removal
    std::vector<int> fixedIdx = findCenterFaceIndices(P, F);

    // Pack everything into the result struct
    ParameterizeResult result;
    result.V           = std::move(V);
    result.F           = std::move(F);
    result.P           = std::move(P);
    result.mesh        = std::move(mesh);
    result.geometry    = std::move(geometry);
    result.MrInv       = std::move(MrInv);
    result.fixedIdx    = std::move(fixedIdx);
    result.scaleFactor = scale;

    return result;
}
