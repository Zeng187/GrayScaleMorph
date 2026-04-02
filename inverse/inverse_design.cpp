#include "inverse_design.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include "boundary_utils.h"
#include "functions.h"
#include "material.hpp"
#include "morphmesh.hpp"
#include "newton.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ---------------------------------------------------------------------------
// Anonymous-namespace helpers
// ---------------------------------------------------------------------------
namespace
{

/// Average a per-vertex field over the three vertices of a face.
double averageVertexDataOnFace(const VertexData<double>& data, Face f)
{
    double sum = 0.0;
    int    cnt = 0;
    for (Vertex v : f.adjacentVertices()) {
        sum += data[v];
        ++cnt;
    }
    return (cnt > 0) ? sum / cnt : 0.0;
}

/// Build a flat 3D position matrix from a 2D parameterization (z = 0).
Eigen::MatrixXd makeFlatFromParam(const Eigen::MatrixXd& P)
{
    Eigen::MatrixXd V_flat = Eigen::MatrixXd::Zero(P.rows(), 3);
    V_flat.col(0) = P.col(0);
    V_flat.col(1) = P.col(1);
    return V_flat;
}

/// Log a stage start.  patch_id < 0 means whole-mesh mode.
void logStageStart(int patch_id, const char* tag, int stage,
                   double wP_kap, double wP_lam)
{
    if (patch_id >= 0)
        spdlog::info("Patch {} Stage {}, {} start, wP_kap: {:.6f}, wP_lam: {:.6f}.",
                     patch_id, stage, tag, wP_kap, wP_lam);
    else
        spdlog::info("Stage {}, {} start, wP_kap: {:.6f}, wP_lam: {:.6f}.",
                     stage, tag, wP_kap, wP_lam);
}

/// Log a stage finish with distance and penalty values.
void logStageFinish(int patch_id, const char* tag, int stage,
                    double distance, double penalty_kap, double penalty_lam)
{
    if (patch_id >= 0)
        spdlog::info("Patch {} Stage {}, {} finish - Distance: {:.6f}, "
                     "Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                     patch_id, stage, tag, distance, penalty_kap, penalty_lam);
    else
        spdlog::info("Stage {}, {} finish- Distance: {:.6f}, "
                     "Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                     stage, tag, distance, penalty_kap, penalty_lam);
}

/// Log projected distance after each stage.
void logProjectedDistance(int patch_id, int stage, double dist_proj)
{
    if (patch_id >= 0)
        spdlog::info("Patch {} Stage {}, Projected distance: {:.6f}",
                     patch_id, stage, dist_proj);
    else
        spdlog::info("Stage {}, Projected distance: {:.6f}", stage, dist_proj);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

InverseDesignResult runInverseDesign(const InverseDesignProblem& prob)
{
    // -- Validate ----------------------------------------------------------
    if (!prob.mesh || !prob.geometry || !prob.ac)
        throw std::invalid_argument(
            "runInverseDesign: mesh, geometry, and ac must be non-null.");

    // Hardcoded material constants (dimensionless reference stiffness).
    constexpr double E  = 1.0;
    constexpr double nu = 0.5;

    ManifoldSurfaceMesh&    mesh     = *prob.mesh;
    VertexPositionGeometry& geometry = *prob.geometry;
    const ActiveComposite&  ac       = *prob.ac;

    // Build boundary-face reference mapping for shape operator computation.
    std::vector<bool> is_boundary;
    std::vector<int> ref_faces = buildRefFaces(mesh, is_boundary);

    const int nV = static_cast<int>(prob.V.rows());
    const int nF = static_cast<int>(prob.F.rows());

    // =====================================================================
    // 1. Compute target morphing parameters (lambda, kappa)
    // =====================================================================
    Morphmesh morph(prob.V, prob.P, prob.F, E, nu);

    Morphmesh::ComputeMorphophing(
        geometry, prob.V, prob.F, nV, nF,
        prob.MrInv,
        morph.lambda_pv_t, morph.lambda_pf_t,
        morph.kappa_pv_t,  morph.kappa_pf_t,
        &morph.vertex_area_sum);

    Morphmesh::SetMorphophing(
        morph.lambda_pv_t, morph.lambda_pf_t,
        morph.kappa_pv_t,  morph.kappa_pf_t,
        morph.lambda_pv_s, morph.lambda_pf_s,
        morph.kappa_pv_s,  morph.kappa_pf_s);

    // =====================================================================
    // 2. Wrap morph parameters into geometry-central containers
    // =====================================================================
    FaceData<double>   lambda_pf_s(mesh, morph.lambda_pf_s);
    VertexData<double> kappa_pv_s (mesh, morph.kappa_pv_s);

    if (prob.patch_id >= 0)
        spdlog::info("Patch {}: Starting SGN inverse design.", prob.patch_id);
    else
        spdlog::info("Starting SGN inverse design.");

    // =====================================================================
    // 3. SGN alternating optimisation loop (up to 5 stages)
    // =====================================================================
    const Eigen::MatrixXd targetV = prob.V;
    Eigen::MatrixXd Vr = prob.V;   // current deformed shape, initialised to target

    double wP_kap = prob.wP_kap;
    double wP_lam = prob.wP_lam;
    double wM_kap = prob.wM_kap;
    double wL_kap = prob.wL_kap;
    double wM_lam = prob.wM_lam;
    double wL_lam = prob.wL_lam;

    const double wM_kap_init = wM_kap;
    const double wL_kap_init = wL_kap;
    const double wM_lam_init = wM_lam;
    const double wL_lam_init = wL_lam;

    constexpr int kMaxStages = 5;

    double distance    = 0.0;
    double penalty_kap = 0.0;
    double penalty_lam = 0.0;

    for (int k = 0; k < kMaxStages; ++k) {
        // ----- OptKap: optimise kappa (per-vertex), fix lambda (per-face) ----
        auto penalty_to_kapp = JointMaterialPenaltyPerV_OptKap(
            geometry, prob.F, lambda_pf_s,
            ac.feasible_lamb, ac.feasible_kapp, prob.betaP);

        logStageStart(prob.patch_id, "OptKap", k, wP_kap, wP_lam);

        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(
            geometry, prob.F, prob.MrInv, lambda_pf_s,
            E, nu, ac.thickness, prob.w_s, prob.w_b, ref_faces);

        Vr = sparse_gauss_newton_FixLam_OptKap_Penalty(
            geometry, targetV, Vr, prob.MrInv,
            lambda_pf_s, kappa_pv_s,
            adjointFunc_OptKap, penalty_to_kapp, prob.fixedIdx,
            prob.max_iter, prob.epsilon, wM_kap, wL_kap, wP_kap,
            E, nu, ac.thickness, prob.w_s, prob.w_b, ref_faces);

        distance    = (Vr - targetV).squaredNorm() / nV;
        penalty_kap = compute_candidate_diff(ac.feasible_kapp, kappa_pv_s.toVector(), true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
        logStageFinish(prob.patch_id, "OptKap", k, distance, penalty_kap, penalty_lam);

        // ----- OptLam: optimise lambda (per-face), fix kappa (per-vertex) ----
        auto penalty_to_lamb = JointMaterialPenaltyPerF_OptLam(
            geometry, prob.F, kappa_pv_s,
            ac.feasible_lamb, ac.feasible_kapp, prob.betaP);

        logStageStart(prob.patch_id, "OptLam", k, wP_kap, wP_lam);

        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(
            geometry, prob.F, prob.MrInv, kappa_pv_s,
            E, nu, ac.thickness, prob.w_s, prob.w_b, ref_faces);

        Vr = sparse_gauss_newton_FixKap_OptLam_Penalty(
            geometry, targetV, Vr, prob.MrInv,
            lambda_pf_s, kappa_pv_s,
            adjointFunc_OptLam, penalty_to_lamb, prob.fixedIdx,
            prob.max_iter, prob.epsilon, wM_lam, wL_lam, wP_lam,
            E, nu, ac.thickness, prob.w_s, prob.w_b, ref_faces);

        distance    = (Vr - targetV).squaredNorm() / nV;
        penalty_kap = compute_candidate_diff(ac.feasible_kapp, kappa_pv_s.toVector(), true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
        logStageFinish(prob.patch_id, "OptLam", k, distance, penalty_kap, penalty_lam);

        // ----- Evaluate projected distance (within the loop) -----------------
        {
            FaceData<double> kappa_pf_proj(mesh);
            FaceData<double> lambda_pf_proj(mesh);

            for (Face f : mesh.faces()) {
                double kap = averageVertexDataOnFace(kappa_pv_s, f);
                double lam = lambda_pf_s[f];
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap, lam);
                kappa_pf_proj[f]  = ac.feasible_kapp[idx];
                lambda_pf_proj[f] = ac.feasible_lamb[idx];
            }

            auto simFunc_proj = simulationFunction(
                geometry, prob.MrInv, lambda_pf_proj, kappa_pf_proj,
                E, nu, ac.thickness, prob.w_s, prob.w_b, ref_faces);

            // Start from flat state to match physical initial condition.
            Eigen::MatrixXd Vr_proj = makeFlatFromParam(prob.P);
            newton(geometry, Vr_proj, simFunc_proj,
                   prob.max_iter, prob.epsilon, false, prob.fixedIdx);

            double dist_proj = (Vr_proj - targetV).squaredNorm() / nV;
            logProjectedDistance(prob.patch_id, k, dist_proj);
        }

        // ----- Update penalty / regularisation weights -----------------------
        if (penalty_kap >= prob.penalty_threshold)
            wP_kap *= 2.0;
        if (penalty_lam >= prob.penalty_threshold)
            wP_lam *= 2.0;

        wM_kap = std::max(wM_kap * 0.5, wM_kap_init * 1e-3);
        wL_kap = std::max(wL_kap * 0.5, wL_kap_init * 1e-3);
        wM_lam = std::max(wM_lam * 0.5, wM_lam_init * 1e-3);
        wL_lam = std::max(wL_lam * 0.5, wL_lam_init * 1e-3);

        if (penalty_kap < prob.penalty_threshold &&
            penalty_lam < prob.penalty_threshold)
            break;
    }

    // =====================================================================
    // 4. Material projection to feasible (t1, t2) per face
    // =====================================================================
    if (prob.patch_id >= 0)
        spdlog::info("Patch {}: Material projection + forward verification.", prob.patch_id);
    else
        spdlog::info("Material projection + forward verification.");

    InverseDesignResult result;
    result.V_inv      = Vr;
    result.t1         = Eigen::VectorXd::Zero(nF);
    result.t2         = Eigen::VectorXd::Zero(nF);
    result.lam_excess = Eigen::VectorXd::Zero(nF);
    result.kap_excess = Eigen::VectorXd::Zero(nF);

    FaceData<double> kappa_pf_final(mesh);
    FaceData<double> lambda_pf_final(mesh);

    for (Face f : mesh.faces()) {
        const int fid = static_cast<int>(f.getIndex());

        const double kap_f = averageVertexDataOnFace(kappa_pv_s, f);
        const double lam_f = lambda_pf_s[f];

        int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap_f, lam_f);
        kappa_pf_final[f]  = ac.feasible_kapp[idx];
        lambda_pf_final[f] = ac.feasible_lamb[idx];
        result.t1[fid]     = ac.feasible_t_vals[idx].first;
        result.t2[fid]     = ac.feasible_t_vals[idx].second;

        result.lam_excess[fid] = std::max(0.0,
            std::max(lam_f - ac.range_lam.y, ac.range_lam.x - lam_f));
        result.kap_excess[fid] = std::max(0.0,
            std::max(kap_f - ac.range_kap.y, ac.range_kap.x - kap_f));
    }

    // =====================================================================
    // 5. Forward verification from flat initial state
    // =====================================================================
    auto simFunc_final = simulationFunction(
        geometry, prob.MrInv, lambda_pf_final, kappa_pf_final,
        E, nu, ac.thickness, prob.w_s, prob.w_b, ref_faces);

    result.V_proj = makeFlatFromParam(prob.P);
    newton(geometry, result.V_proj, simFunc_final,
           prob.max_iter, prob.epsilon, true, prob.fixedIdx);

    // =====================================================================
    // 6. Compute error metrics
    // =====================================================================
    result.dist_inv  = (result.V_inv  - targetV).squaredNorm() / nV;
    result.dist_proj = (result.V_proj - targetV).squaredNorm() / nV;

    if (prob.patch_id >= 0)
        spdlog::info("Patch {}: Final projected distance: {:.6f}",
                     prob.patch_id, result.dist_proj);
    else
        spdlog::info("Final projected distance: {:.6f}", result.dist_proj);

    return result;
}
