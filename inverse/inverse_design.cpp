#include "inverse_design.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#include "boundary_utils.h"
#include "functions.h"
#include "material.hpp"
#include "morphmesh.hpp"
#include "newton.h"
#include "rigid_align.h"

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

/// Log projected distance after each phase (highlighted + trailing blank line).
void logProjectedDistance(int patch_id, int stage, const std::string& phase, double dist_proj)
{
    if (patch_id >= 0)
        spdlog::info("\033[1;33m>>> Patch {} Stage {} {}, Projected distance: {:.6f} <<<\033[0m",
                     patch_id, stage, phase, dist_proj);
    else
        spdlog::info("\033[1;33m>>> Stage {} {}, Projected distance: {:.6f} <<<\033[0m",
                     stage, phase, dist_proj);
    fmt::print("\n");
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

    // Poisson ratio comes from Resources/setup/global.json via
    // InverseDesignProblem::poisson_ratio. Young's modulus is per-face (see
    // E_face below) and driven by the material curve E(t1, t2). Both are
    // dimensionless: E is normalised by E_ref so that the numerical regime
    // matches the legacy scalar-E=1 code.
    const double nu = prob.poisson_ratio;

    ManifoldSurfaceMesh&    mesh     = *prob.mesh;
    VertexPositionGeometry& geometry = *prob.geometry;
    const ActiveComposite&  ac       = *prob.ac;

    const double E_ref = referenceModulus(ac);

    // Build per-face modulus by projecting the current continuous (lambda, kappa)
    // onto the nearest feasible (t1, t2) and reading the corresponding E from the
    // feasible set (already normalised by E_ref). This is the lagged/frozen-E
    // scheme: within a Newton stage E_face is treated as constant; it is only
    // refreshed between stages so each Gauss-Newton step still sees a stationary
    // quadratic model.
    // Strict Saint-Venant elastic energy distance with target-metric area
    // element sqrt(det(gbar_j)) = lambdabar_j^2:
    //   d^2 = lambdabar_j^2 * [ a * (lambda^2 - lambdabar_j^2)^2
    //                         + b * (kappa - kappabar_j)^2 ]
    // a = h/4, b = h^3/12.  E_face and 1/(1-nu) are common to all candidates
    // and cancel in argmin -- kept implicit here.  Mirrors the metric inside
    // `JointMaterialPenaltyPerF_*` in functions.cpp so the continuous-time
    // gradient and the discrete projection use the same Voronoi tessellation.
    const double a_unit_sv = ac.thickness / 4.0;
    const double b_unit_sv = ac.thickness * ac.thickness * ac.thickness / 12.0;
    auto findFeasibleEnergy = [&](double kap, double lam) -> int {
        const double lam_sq = lam * lam;
        int    best_i = 0;
        double best_d = std::numeric_limits<double>::max();
        for (size_t i = 0; i < ac.feasible_lamb.size(); ++i) {
            const double feas_lam_sq = ac.feasible_lamb[i] * ac.feasible_lamb[i];
            const double dlsq = lam_sq - feas_lam_sq;
            const double dk   = kap - ac.feasible_kapp[i];
            const double d    = feas_lam_sq * (a_unit_sv * dlsq * dlsq
                                              + b_unit_sv * dk * dk);
            if (d < best_d) { best_d = d; best_i = static_cast<int>(i); }
        }
        return best_i;
    };

    auto buildEFaceFromState = [&](const FaceData<double>& /*lam_pf*/,
                                   const FaceData<double>& /*kap_pf*/) -> FaceData<double> {
        // DEBUG: uniform E=1.  Mirrors the Forward override so twin-experiment
        // Verify isolates pure (lambda, kappa) recovery without the
        // continuous-poly-vs-discrete-grid E mismatch.  Restore the
        // commented-out projection block below to re-enable curve-driven E.
        FaceData<double> E_face(mesh);
        for (Face f : mesh.faces()) {
            E_face[f] = 1.0;
            // const int idx = findFeasibleEnergy(kap_pf[f], lam_pf[f]);
            // E_face[f] = ac.feasible_modl[idx] / E_ref;
        }
        return E_face;
    };

    // Build boundary-face reference mapping for shape operator computation.
    std::vector<bool> is_boundary;
    std::vector<int> ref_faces = buildRefFaces(mesh, is_boundary);

    const int nV = static_cast<int>(prob.V.rows());
    const int nF = static_cast<int>(prob.F.rows());

    // =====================================================================
    // 1. Compute target morphing parameters (lambda, kappa)
    // =====================================================================
    // Morphmesh here is used purely as a container for (lambda, kappa) fields
    // via ComputeMorphophing / SetMorphophing — its scalar E/nu slots are
    // only consumed by ComputeElasticEnergy, which this pipeline does not call.
    // Passing 1.0 / nu keeps the ctor happy without affecting results.
    Morphmesh morph(prob.V, prob.P, prob.F, 1.0, nu);

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

    // Oracle override: when an exact (lambda, kappa) ground truth is supplied
    // (e.g., recomputed from the original (t1, t2) of a twin experiment),
    // bypass the Morphophing-from-V step which always carries a thin-shell
    // realisability residual.  We override only the per-face state used
    // downstream by the SGN inner solves; per-vertex lambda/kappa are dead
    // weight in this pipeline.
    if (prob.oracle_lambda_pf.size() == nF && prob.oracle_kappa_pf.size() == nF) {
        spdlog::info(
            "Oracle init: replacing Morphophing-from-V with user-supplied (lambda, kappa) "
            "(per-face), nF={}.",
            nF);
        morph.lambda_pf_t = prob.oracle_lambda_pf;
        morph.kappa_pf_t  = prob.oracle_kappa_pf;
        morph.lambda_pf_s = prob.oracle_lambda_pf;
        morph.kappa_pf_s  = prob.oracle_kappa_pf;
    }

    // =====================================================================
    // 2. Wrap morph parameters into geometry-central containers
    // =====================================================================
    FaceData<double> lambda_pf_s(mesh, morph.lambda_pf_s);
    FaceData<double> kappa_pf_s (mesh, morph.kappa_pf_s);

    if (prob.patch_id >= 0)
        spdlog::info("Patch {}: Starting SGN inverse design.", prob.patch_id);
    else
        spdlog::info("Starting SGN inverse design.");

    // -------------------------------------------------------------------
    // Trajectory CSV setup (optional diagnostic).
    // -------------------------------------------------------------------
    std::ofstream inner_csv;
    std::ofstream outer_csv;
    if (!prob.trajectory_dir.empty()) {
        std::filesystem::create_directories(prob.trajectory_dir);
        const std::string base = prob.trajectory_dir + "/" + prob.trajectory_tag;
        inner_csv.open(base + "_inner.csv");
        outer_csv.open(base + "_outer.csv");
        if (inner_csv.is_open()) {
            inner_csv << "stage,phase,iter,wM,wL,wP,beta,"
                         "data_fit,wM_term,wL_term,wP_term,total_J,"
                         "decrement,step_size\n";
        }
        if (outer_csv.is_open()) {
            outer_csv << "stage,phase,wM_kap,wL_kap,wP_kap,wM_lam,wL_lam,wP_lam,beta,"
                         "dist_inv,penalty_kap,penalty_lam,dist_proj\n";
        }
        spdlog::info("Trajectory CSV: {}_inner.csv / _outer.csv", base);
    }

    // =====================================================================
    // 3. SGN alternating optimisation loop (up to 5 stages)
    // =====================================================================
    const Eigen::MatrixXd targetV = prob.V;
    // SGN's adjoint-based OptKap / OptLam are formulated around a small
    // residual ||V - targetV||, so the continuous solve must keep Vr near
    // target.  Flat-start initialisation (codex's original suggestion) makes
    // the residual O(target-shape) at iteration 0 and the Newton bridge
    // fails -- empirically dist_inv jumps to ~200.
    //
    // Branch mismatch between this target-warm-start Vr and the cold
    // flat-start verification is mitigated by warm-starting the in-loop
    // diagnostic from Vr (see Vr_proj below) and by keeping the final
    // manufacturable verification flat-start (multi-stability is then a
    // visible physical signal, not an algorithm artefact).
    Eigen::MatrixXd Vr = prob.V;

    // Per-phase projected verification: project current continuous (lambda,
    // kappa) onto the feasible grid (idx-by-`findFeasibleEnergy`), forward-
    // simulate from the warm Vr starting point, and report the rigid-aligned
    // distance to the target.  Called once after OptKap and once after
    // OptLam so the trajectory CSV captures dist_proj on every phase row.
    auto compute_dist_proj = [&]() -> double {
        FaceData<double> kappa_pf_proj(mesh);
        FaceData<double> lambda_pf_proj(mesh);
        FaceData<double> E_face_proj(mesh);

        for (Face f : mesh.faces()) {
            int idx = findFeasibleEnergy(kappa_pf_s[f], lambda_pf_s[f]);
            kappa_pf_proj[f]  = ac.feasible_kapp[idx];
            lambda_pf_proj[f] = ac.feasible_lamb[idx];
            E_face_proj[f]    = 1.0;
        }

        auto simFunc_proj = simulationFunction(
            geometry, prob.MrInv, lambda_pf_proj, kappa_pf_proj,
            E_face_proj, nu, ac.thickness, prob.w_s, prob.w_b, ref_faces);

        Eigen::MatrixXd Vr_proj = Vr;
        newton(geometry, Vr_proj, simFunc_proj,
               prob.verify_max_iter, prob.verify_epsilon, false, prob.fixedIdx);

        Vr_proj = rigidAlign(Vr_proj, targetV);
        return (Vr_proj - targetV).rowwise().norm().mean();
    };

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

    const int kMaxStages = prob.max_stages;

    double distance    = 0.0;
    double penalty_kap = 0.0;
    double penalty_lam = 0.0;

    for (int k = 0; k < kMaxStages; ++k) {
        // ----- OptKap: optimise kappa (per-vertex), fix lambda (per-face) ----
        FaceData<double> E_face_kap = buildEFaceFromState(lambda_pf_s, kappa_pf_s);

        auto penalty_to_kapp = JointMaterialPenaltyPerF_OptKap(
            geometry, prob.F, lambda_pf_s, E_face_kap,
            ac.feasible_lamb, ac.feasible_kapp,
            ac.thickness, nu, prob.well_scale);

        logStageStart(prob.patch_id, "OptKap", k, wP_kap, wP_lam);

        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(
            geometry, prob.F, prob.MrInv, lambda_pf_s,
            E_face_kap, nu, ac.thickness, prob.w_s, prob.w_b, ref_faces);

        InnerIterLog log_kap{
            inner_csv.is_open() ? &inner_csv : nullptr, k, "OptKap", prob.well_scale};
        Vr = sparse_gauss_newton_FixLam_OptKap_Penalty(
            geometry, targetV, Vr, prob.MrInv,
            lambda_pf_s, kappa_pf_s,
            adjointFunc_OptKap, penalty_to_kapp, prob.fixedIdx,
            prob.max_iter, prob.epsilon, wM_kap, wL_kap, wP_kap,
            E_face_kap, nu, ac.thickness, prob.w_s, prob.w_b, ref_faces,
            [](const Eigen::VectorXd&) {}, log_kap);

        distance    = (Vr - targetV).rowwise().norm().mean();
        penalty_kap = compute_candidate_diff(ac.feasible_kapp, kappa_pf_s.toVector(), true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
        logStageFinish(prob.patch_id, "OptKap", k, distance, penalty_kap, penalty_lam);

        const double dist_proj_kap = compute_dist_proj();
        logProjectedDistance(prob.patch_id, k, "OptKap", dist_proj_kap);

        if (outer_csv.is_open()) {
            outer_csv << k << ",OptKap,"
                      << wM_kap << "," << wL_kap << "," << wP_kap << ","
                      << wM_lam << "," << wL_lam << "," << wP_lam << ","
                      << prob.well_scale << ","
                      << distance << "," << penalty_kap << "," << penalty_lam << ","
                      << dist_proj_kap << "\n";
        }

        // ----- OptLam: optimise lambda (per-face), fix kappa (per-vertex) ----
        FaceData<double> E_face_lam = buildEFaceFromState(lambda_pf_s, kappa_pf_s);

        auto penalty_to_lamb = JointMaterialPenaltyPerF_OptLam(
            geometry, prob.F, kappa_pf_s, E_face_lam,
            ac.feasible_lamb, ac.feasible_kapp,
            ac.thickness, nu, prob.well_scale);

        logStageStart(prob.patch_id, "OptLam", k, wP_kap, wP_lam);

        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(
            geometry, prob.F, prob.MrInv, kappa_pf_s,
            E_face_lam, nu, ac.thickness, prob.w_s, prob.w_b, ref_faces);

        InnerIterLog log_lam{
            inner_csv.is_open() ? &inner_csv : nullptr, k, "OptLam", prob.well_scale};
        Vr = sparse_gauss_newton_FixKap_OptLam_Penalty(
            geometry, targetV, Vr, prob.MrInv,
            lambda_pf_s, kappa_pf_s,
            adjointFunc_OptLam, penalty_to_lamb, prob.fixedIdx,
            prob.max_iter, prob.epsilon, wM_lam, wL_lam, wP_lam,
            E_face_lam, nu, ac.thickness, prob.w_s, prob.w_b, ref_faces,
            [](const Eigen::VectorXd&) {}, log_lam);

        distance    = (Vr - targetV).rowwise().norm().mean();
        penalty_kap = compute_candidate_diff(ac.feasible_kapp, kappa_pf_s.toVector(), true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
        logStageFinish(prob.patch_id, "OptLam", k, distance, penalty_kap, penalty_lam);

        const double dist_proj_lam = compute_dist_proj();
        logProjectedDistance(prob.patch_id, k, "OptLam", dist_proj_lam);

        if (outer_csv.is_open()) {
            outer_csv << k << ",OptLam,"
                      << wM_kap << "," << wL_kap << "," << wP_kap << ","
                      << wM_lam << "," << wL_lam << "," << wP_lam << ","
                      << prob.well_scale << ","
                      << distance << "," << penalty_kap << "," << penalty_lam << ","
                      << dist_proj_lam << "\n";
        }

        // ----- Update penalty / regularisation weights -----------------------
        if (penalty_kap >= prob.penalty_threshold)
            wP_kap *= prob.wP_growth_factor;
        if (penalty_lam >= prob.penalty_threshold)
            wP_lam *= prob.wP_growth_factor;

        wM_kap = std::max(wM_kap * prob.wM_decay_factor, wM_kap_init * 1e-3);
        wL_kap = std::max(wL_kap * prob.wL_decay_factor, wL_kap_init * 1e-3);
        wM_lam = std::max(wM_lam * prob.wM_decay_factor, wM_lam_init * 1e-3);
        wL_lam = std::max(wL_lam * prob.wL_decay_factor, wL_lam_init * 1e-3);

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
    FaceData<double> E_face_final(mesh);

    for (Face f : mesh.faces()) {
        const int fid = static_cast<int>(f.getIndex());

        const double kap_f = kappa_pf_s[f];
        const double lam_f = lambda_pf_s[f];

        int idx = findFeasibleEnergy(kap_f, lam_f);
        kappa_pf_final[f]  = ac.feasible_kapp[idx];
        lambda_pf_final[f] = ac.feasible_lamb[idx];
        E_face_final[f]    = 1.0;  // DEBUG: uniform E (see buildEFaceFromState)
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
        E_face_final, nu, ac.thickness, prob.w_s, prob.w_b, ref_faces);

    result.V_proj = makeFlatFromParam(prob.P);
    newton(geometry, result.V_proj, simFunc_final,
           prob.verify_max_iter, prob.verify_epsilon, true, prob.fixedIdx);

    // =====================================================================
    // 6. Rigid-align V_proj to target, then compute error metrics
    // =====================================================================
    //    V_proj is anchored at the flat-plate center (z=0), while targetV
    //    is the 3D target shape — they differ by a rigid body transform.
    //    Align first so that dist_proj reflects only the design error.
    result.V_proj = rigidAlign(result.V_proj, targetV);

    result.dist_inv  = (result.V_inv  - targetV).rowwise().norm().mean();
    result.dist_proj = (result.V_proj - targetV).rowwise().norm().mean();

    if (prob.patch_id >= 0) {
        spdlog::info("Patch {}: Final inverse distance: {:.6f}", prob.patch_id, result.dist_inv);
        spdlog::info("\033[1;33m>>> Patch {}: Final projected distance (aligned): {:.6f} <<<\033[0m",
                     prob.patch_id, result.dist_proj);
    } else {
        spdlog::info("Final inverse distance: {:.6f}", result.dist_inv);
        spdlog::info("\033[1;33m>>> Final projected distance (aligned): {:.6f} <<<\033[0m",
                     result.dist_proj);
    }

    if (outer_csv.is_open()) {
        outer_csv << kMaxStages << ",Final,"
                  << wM_kap << "," << wL_kap << "," << wP_kap << ","
                  << wM_lam << "," << wL_lam << "," << wP_lam << ","
                  << prob.well_scale << ","
                  << result.dist_inv << ",,," << result.dist_proj << "\n";
    }

    return result;
}
