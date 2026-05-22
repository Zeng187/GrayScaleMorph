// Inverse: inverse design on a SINGLE mesh (treated as patch_0).
//
// Reads:
//   PathSetting.TargetDir + {model}/patch_0_V.obj             (final physical-unit target)
//   PathSetting.ParamDir  + {model}/patch_0_P.obj             (scaled + gauge-shifted P)
//   PathSetting.CondDir   + {model}/patch_0_bound_center.txt  (3 vertex idx)
//
// Writes:
//   PathSetting.MorphDir  + {model}/patch_0_targ.obj     (= V; side-by-side w/ inv)
//   PathSetting.MorphDir  + {model}/patch_0_inv.obj      (Vr, SGN continuous optimum)
//   PathSetting.DesignDir + {model}/patch_0_material.txt (face_id t1 t2)
//
// V_target is already physical-unit (Param has applied globalScale), so no
// inverse-rescaling here.  Run Param first.

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/loop.h>
#include <igl/cotmatrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <filesystem>

#include <spdlog/spdlog.h>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include "config.hpp"
#include "material.hpp"
#include "parameterization.h"
#include "simulation_utils.h"
#include "functions.h"
#include "newton.h"
#include "LocalGlobalSolver.h"
#include "morphmesh.hpp"
#include "morph_functions.hpp"
#include "boundary_utils.h"

#define __Add_PENALTY__

int main(int /*argc*/, char * /*argv*/[])
{
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    Config config("cfg.json");
    ActiveComposite ac(config.materialJsonPath());
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    spdlog::info("Inverse (single mesh): start.");

    const std::string model = config.ModelSetting.ModelName;
    const std::string target_dir = config.PathSetting.TargetDir + model + "/";
    const std::string param_dir = config.PathSetting.ParamDir + model + "/";
    const std::string cond_dir = config.PathSetting.CondDir + model + "/";
    const std::string v_path = target_dir + "patch_0_V.obj";
    const std::string p_path = param_dir + "patch_0_P.obj";
    const std::string morph_dir = config.PathSetting.MorphDir + model + "/";
    const std::string design_dir = config.PathSetting.DesignDir + model + "/";
    std::filesystem::create_directories(morph_dir);
    std::filesystem::create_directories(design_dir);

    // -------- Load target V (already scaled by Param) --------
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::readOBJ(v_path, V, F))
    {
        spdlog::error("Cannot read target V: {}.  Run Param first.", v_path);
        return -1;
    }
    size_t nV = V.rows();
    size_t nF = F.rows();
    spdlog::info("Target V: {} V, {} F (from {}).", nV, nF, v_path);

    // -------- Load P + scale from Param's output --------
    Eigen::MatrixXd P_loaded3;
    Eigen::MatrixXi F_p;
    if (!igl::readOBJ(p_path, P_loaded3, F_p))
    {
        spdlog::error("Cannot read parameterisation: {}.  Run Param first.", p_path);
        return -1;
    }
    if (P_loaded3.rows() != (Eigen::Index)nV)
    {
        spdlog::error("P vertex count ({}) does not match V count ({}).",
                      P_loaded3.rows(), nV);
        return -1;
    }
    Eigen::MatrixXd P(nV, 2);
    P = P_loaded3.leftCols(2);

    ManifoldSurfaceMesh mesh(F);
    VertexPositionGeometry geometry(mesh, V);
    geometry.refreshQuantities();

    std::vector<bool> is_boundary_face;
    std::vector<int> ref_faces = buildRefFaces(mesh, is_boundary_face);

    // -------- Setup target morphing parameters --------
    spdlog::info("Step 2: Material Settings.");
    Eigen::MatrixXd targetV = V;
    FaceData<Eigen::MatrixXd> M = precomputeM(mesh, V, F);
    FaceData<Eigen::Matrix2d> MrInv = precomputeMrInv(mesh, P, F);

    // Boundary condition: read 3 vertex indices from cond file written by Param
    std::vector<int> fixedVertexIdx;
    {
        const std::string cond_path = cond_dir + "patch_0_bound_center.txt";
        std::ifstream ifs(cond_path);
        if (!ifs.is_open())
        {
            spdlog::error("Cannot read cond file: {}.  Run Param first.", cond_path);
            return -1;
        }
        int v0, v1, v2;
        ifs >> v0 >> v1 >> v2;
        fixedVertexIdx = {v0, v1, v2};
        spdlog::info("Loaded cond: v0={} v1={} v2={}", v0, v1, v2);
    }
    std::vector<int> fixedIdx;
    for (int v : fixedVertexIdx)
        for (int k = 0; k < 3; ++k)
            fixedIdx.push_back(3 * v + k);
    std::sort(fixedIdx.begin(), fixedIdx.end());

    double E = 1.0;
    double nu = 0.5;
    Morphmesh morph_mesh(V, P, F, E, nu);
    Morphmesh::ComputeMorphophing(geometry, V, F, nV, nF, ref_faces,
                                  MrInv, morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t, morph_mesh.kappa_pv_t, morph_mesh.kappa_pf_t, &morph_mesh.vertex_area_sum);
    Morphmesh::SetMorphophing(morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t,
                              morph_mesh.kappa_pv_t, morph_mesh.kappa_pf_t,
                              morph_mesh.lambda_pv_s, morph_mesh.lambda_pf_s,
                              morph_mesh.kappa_pf_s, morph_mesh.kappa_pf_s);

    // Mirror the target into outputs/ so {targ, inv} sit side-by-side for
    // easy comparison.  Content is the same as 2_target/.../patch_0_V.obj.
    igl::writeOBJ(morph_dir + "patch_0_targ.obj", V, F);

    VertexData<double> lambda_pv_s(mesh, morph_mesh.lambda_pv_s);
    FaceData<double> lambda_pf_s(mesh, morph_mesh.lambda_pf_s);
    FaceData<double> kappa_pf_s(mesh, morph_mesh.kappa_pf_s);

    // V_init: flat plate (P embedded as z=0) rigidly aligned so the 3 fixed
    // vertices sit exactly at their target positions in V.
    Eigen::MatrixXd Vr = flatPlateAligned(P, V, fixedVertexIdx);
    igl::writeOBJ(morph_dir + "patch_0_init.obj", Vr, F);

    // Lumped vertex mass vector (size 3*nV); shared by every SGN call so the
    // distance/SPN values reported by main and inside the solver are computed
    // from the exact same weights.
    const Eigen::VectorXd masses = computeVertexMasses(geometry);

    // Face-space matrices used to build the *other-variable* regulariser as a
    // constant offset, so the SPN energy printed by the OptKap/OptLam stages
    // share the same formula:  distance + kappa_reg + lambda_reg.
    // M_kappa depends on MrInv -> must be refreshed if P (and hence MrInv)
    // changes between stages (lambda-aware ARAP P-update).
    Eigen::SparseMatrix<double> M_kappa = computeFaceMassKappa(mesh, MrInv);
    const Eigen::SparseMatrix<double> M_lambda = computeFaceMassLambda(geometry);
    const Eigen::SparseMatrix<double> L_face = computeFaceDualLaplacian(mesh);

    // ARAP solver kept around for optional warm-up / fallback (not used by
    // the SGN OptP path).
    LocalGlobalSolver paramSolver(V, F);

    // ---- P-side regularisation matrices (size 2|V| x 2|V|) -----------------
    // M_P_2: per-vertex P-mass (identity).  Used to define ||P - P_anchor||^2
    // L_P_2: per-vertex P-Laplacian (cotan L on V, kron'd with I_2).
    // Both stay constant across stages (topology never changes).
    Eigen::SparseMatrix<double> M_P_2(2 * nV, 2 * nV);
    M_P_2.setIdentity();
    Eigen::SparseMatrix<double> L_P_2(2 * nV, 2 * nV);
    {
        // Build a per-vertex cotan Laplacian from libigl on V then kron with I_2.
        Eigen::SparseMatrix<double> L_v;
        igl::cotmatrix(V, F, L_v);
        L_v = (-L_v).eval();  // libigl uses negative-Laplacian convention -> flip
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(L_v.nonZeros() * 2);
        for (int i = 0; i < L_v.outerSize(); ++i)
            for (Eigen::SparseMatrix<double>::InnerIterator it(L_v, i); it; ++it)
                for (int d = 0; d < 2; ++d)
                    trips.emplace_back(2 * (int)it.row() + d, 2 * (int)it.col() + d, it.value());
        L_P_2.setFromTriplets(trips.begin(), trips.end());
    }

    // P_anchor: initial parameterisation from ParamAll, used as the reference
    // for the ||P - P_anchor||^2 regulariser inside SGN OptP.
    const Eigen::MatrixXd P_anchor = P;

    spdlog::info("Step 4: Inverse Design.");

    double wP_lam = config.RuntimeSetting.wP_lam;
    double wP_kap = config.RuntimeSetting.wP_kap;
    double penalty_threshold = config.RuntimeSetting.penalty_threshold;
    double betaP = config.RuntimeSetting.betaP;
    // 1D independent hard-min penalties: kappa pulls toward the nearest 1D
    // kappa candidate; lambda pulls toward the nearest 1D lambda candidate
    // (argmins can pick different candidate indices).  Empirically gives a
    // smaller proj_dist than 2D joint variants on the hemisphere benchmark
    // -- the looser pull lets distance keep optimising in the (lambda, kappa)
    // gaps between candidate pairs.  wP_kap / wP_lam control the per-direction
    // penalty weight at the SGN call (not baked into the penalty function).
    auto penalty_to_kapp = MaterialPenaltyFunctionPerF(geometry, ac.feasible_kapp, betaP);
    auto penalty_to_lamb = MaterialPenaltyFunctionPerF(geometry, ac.feasible_lamb, betaP);

    int stage_iter = config.RuntimeSetting.stage_iter;
    int k = 0;

    double wM_kap = config.RuntimeSetting.wM_kap;
    double wM_lam = config.RuntimeSetting.wM_lam;
    double wL_kap = config.RuntimeSetting.wL_kap;
    double wL_lam = config.RuntimeSetting.wL_lam;

    double distance = 0.0;
    double spn_energy = 0.0;
    double penalty_kap = 0.0;
    double penalty_lam = 0.0;

    // Anchor for the mass-term regulariser: pull theta toward the centre of
    // the feasible candidate set, not toward 0.  For kappa the candidates are
    // ~symmetric around 0 so kappa_anchor = 0 is fine.  For lambda the
    // candidates are all > 1 (grayscale material can only expand), so anchor
    // at the mean of feasible_lamb avoids the mass-reg pulling lambda away
    // from the feasible region.  Laplacian term is unaffected (L * 1 = 0).
    const double kappa_anchor  = 0.0;
    const double lambda_anchor = std::accumulate(ac.feasible_lamb.begin(),
                                                  ac.feasible_lamb.end(), 0.0)
                                 / static_cast<double>(ac.feasible_lamb.size());
    spdlog::info("Reg anchors: kappa_anchor={}, lambda_anchor={:.6f}",
                 kappa_anchor, lambda_anchor);

    // Regularisation accumulators kept in sync between stages so each SGN call
    // receives the *other* variable's regulariser as a constant offset.
    auto computeKappaReg = [&]()
    {
        const Eigen::VectorXd k = kappa_pf_s.toVector();
        const Eigen::VectorXd ko = k - Eigen::VectorXd::Constant(k.size(), kappa_anchor);
        return wM_kap * ko.dot(M_kappa * ko) + wL_kap * k.dot(L_face * k);
    };
    auto computeLambdaReg = [&]()
    {
        const Eigen::VectorXd l = lambda_pf_s.toVector();
        const Eigen::VectorXd lo = l - Eigen::VectorXd::Constant(l.size(), lambda_anchor);
        return wM_lam * lo.dot(M_lambda * lo) + wL_lam * l.dot(L_face * l);
    };
    double kappa_reg = computeKappaReg();
    double lambda_reg = computeLambdaReg();
    double self_reg = 0.0;

    // Projected distance: snap the current (kappa, lambda) on every face to the
    // nearest feasible material pair, re-run forward Newton from `Vr_start`,
    // return the mass-weighted distance to target.  Vr_start defaults to the
    // current `Vr`; pass a different initial mesh (e.g. SGN iter intermediate)
    // to evaluate the proj_dist at that state.
    // computeProjStateFrom: snap (lambda, kappa) per face to the nearest
    // feasible candidate, run forward Newton from Vr_start, and return BOTH
    // the mass-weighted distance and the resulting Vr_proj.  Callers that
    // need only the distance ignore .V; callers that need per-vertex RMS
    // (boundary / interior) use .V directly.  This avoids a second forward
    // sim per logger invocation.
    struct ProjState { double dist; Eigen::MatrixXd V; };
    auto computeProjStateFrom = [&](const Eigen::MatrixXd& Vr_start) -> ProjState
    {
        FaceData<double> kappa_pf_proj(mesh);
        FaceData<double> lambda_pf_proj(mesh);
        for (Face f : mesh.faces())
        {
            int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb,
                                        kappa_pf_s[f], lambda_pf_s[f]);
            kappa_pf_proj[f]  = ac.feasible_kapp[idx];
            lambda_pf_proj[f] = ac.feasible_lamb[idx];
        }
        auto simFunc_proj = simulationFunction(geometry, MrInv, lambda_pf_proj, kappa_pf_proj,
                                               E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Eigen::MatrixXd Vr_proj = Vr_start;
        newton(geometry, Vr_proj, simFunc_proj,
               config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx);
        double d2 = 0.0;
        for (size_t i = 0; i < nV; ++i)
            for (int j = 0; j < 3; ++j)
            {
                double d = Vr_proj(i, j) - targetV(i, j);
                d2 += masses(3 * i + j) * d * d;
            }
        return { d2, std::move(Vr_proj) };
    };
    auto computeProjectedDistanceFrom = [&](const Eigen::MatrixXd& Vr_start) {
        return computeProjStateFrom(Vr_start).dist;
    };
    auto computeProjectedDistance = [&]() { return computeProjectedDistanceFrom(Vr); };

    // Per-vertex Euclidean RMS distance, separated by boundary / interior.
    // Returned as a pair (bd_rms, int_rms) in physical units.  Unlike the
    // mass-weighted `dist` field this is the raw uniform-averaged RMS, so
    // it has direct geometric meaning ("the average vertex is X mm off
    // target").  bd_rms typically dominates because boundary vertices have
    // fewer geometric constraints than interior ones.
    geometry.requireVertexIndices();
    auto computeBoundaryInteriorRMS = [&](const Eigen::MatrixXd& V_cur) -> std::pair<double, double> {
        double bd_sum2 = 0.0, int_sum2 = 0.0;
        int    bd_cnt  = 0,   int_cnt  = 0;
        for (Vertex v : mesh.vertices()) {
            const int vi = static_cast<int>(geometry.vertexIndices[v]);
            const Eigen::Vector3d d = V_cur.row(vi) - targetV.row(vi);
            const double d2 = d.squaredNorm();
            if (v.isBoundary()) { bd_sum2 += d2; ++bd_cnt; }
            else                { int_sum2 += d2; ++int_cnt; }
        }
        const double bd_rms  = bd_cnt  > 0 ? std::sqrt(bd_sum2 / bd_cnt)   : 0.0;
        const double int_rms = int_cnt > 0 ? std::sqrt(int_sum2 / int_cnt) : 0.0;
        return { bd_rms, int_rms };
    };

    // Re-run forward Newton on the current (P, lambda, kappa) state and
    // recompute mass-weighted distance.  Updates `Vr` in place so the next
    // SGN call starts from the new equilibrium.  Used after the P-update
    // sub-stage where MrInv changed and Vr is no longer in equilibrium.
    auto recomputeForwardState = [&]() -> double
    {
        auto simFunc = simulationFunction(geometry, MrInv, lambda_pf_s, kappa_pf_s,
                                          E, nu, ac.thickness,
                                          config.RuntimeSetting.w_s,
                                          config.RuntimeSetting.w_b, ref_faces);
        newton(geometry, Vr, simFunc,
               config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx);
        double d2 = 0.0;
        for (size_t i = 0; i < nV; ++i)
            for (int j = 0; j < 3; ++j)
            {
                double d = Vr(i, j) - targetV(i, j);
                d2 += masses(3 * i + j) * d * d;
            }
        return d2;
    };

    // One-line stage stats: SPN energy -> Distance -> Projected distance ->
    // boundary / interior per-vertex RMS -> Penalties.
    auto printStageStats = [&]()
    {
        const auto _proj_st = computeProjStateFrom(Vr);
        const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(_proj_st.V);
        penalty_kap = compute_candidate_diff(ac.feasible_kapp, kappa_pf_s.toVector(), true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
        std::cout << "SPN energy: " << spn_energy
                  << ", Distance: " << distance
                  << ", Projected distance: " << _proj_st.dist
                  << ", bd_rms: " << bd_rms
                  << ", int_rms: " << int_rms
                  << ", Penalty_kap: " << penalty_kap
                  << ", Penalty_lam: " << penalty_lam
                  << "\n";
    };

    // ---- Trust-region-style safeguard state -------------------------------
    // Keep a snapshot of the best (lowest dist + proj_dist) state seen so
    // far.  At the end of each stage, if both `distance` and
    // `projected_distance` worsened relative to the best snapshot, the
    // stage is REJECTED: the entire state (Vr, lambda, kappa, P, MrInv,
    // M_kappa, reg accumulators, weights) reverts to the snapshot, and
    // the homotopy growth factor for wP is halved so the next attempt
    // takes a smaller step.  Otherwise the stage is ACCEPTED and the
    // snapshot is updated.
    double dist_best = std::numeric_limits<double>::infinity();
    double proj_best = std::numeric_limits<double>::infinity();
    Eigen::MatrixXd                 Vr_best = Vr;
    FaceData<double>                lambda_pf_best = lambda_pf_s;
    FaceData<double>                kappa_pf_best  = kappa_pf_s;
    Eigen::MatrixXd                 P_best  = P;
    FaceData<Eigen::Matrix2d>       MrInv_best = MrInv;
    Eigen::SparseMatrix<double>     M_kappa_best = M_kappa;
    // wM / wL are homotopy-schedule, not rolled back on REJECT (see above).
    double wP_lam_best = wP_lam;
    double wP_kap_best = wP_kap;
    double kappa_reg_best = kappa_reg, lambda_reg_best = lambda_reg;

    // CSV trajectory log of every SGN iter's (spn, dist) plus end-of-substage
    // markers, written under MorphLogsDir / {method} / {model} / so different
    // morph methods (homotopy, mgda, ste, ...) write to disjoint dirs.
    // Schema:  stage, substage, iter, spn, dist
    // where substage in {OptKap, OptLam, OptP}, iter is the SGN inner-iter
    // index (-1 marks an end-of-substage "final" row).
    const std::string morphlogs_dir = config.PathSetting.MorphLogsDir
                                    + config.RuntimeSetting.morph_method + "/"
                                    + model + "/";
    std::filesystem::create_directories(morphlogs_dir);
    std::ofstream iter_log_ofs(morphlogs_dir + "iter_log.csv");
    iter_log_ofs << "stage,substage,iter,spn,dist,proj_dist,"
                 << "kappa_reg,lambda_reg,penalty_kap,penalty_lam,"
                 << "wP_kap,wP_lam,wM_kap,wL_kap,wM_lam,wL_lam,"
                 << "bd_rms,int_rms\n";
    spdlog::info("Iter log -> {}", morphlogs_dir + "iter_log.csv");

    // Helper to reshape SGN's flat 3*nV x vector back to nV x 3 V matrix.
    auto reshape_x_to_V = [&](const Eigen::VectorXd& x_vec) {
        Eigen::MatrixXd V_iter(nV, 3);
        for (size_t v = 0; v < nV; ++v)
            for (int j = 0; j < 3; ++j)
                V_iter(v, j) = x_vec(3 * v + j);
        return V_iter;
    };

    // Independent homotopy growth factors for wP_lam (A) and wP_kap (B).
    // wP_lam_new = wP_lam * (1 + wP_lam_growth_factor) ; similarly wP_kap.
    // Halved on REJECT.  Floored at 1e-4.
    double wP_lam_growth_factor = config.RuntimeSetting.wP_lam_growth_factor;
    double wP_kap_growth_factor = config.RuntimeSetting.wP_kap_growth_factor;

    while (k < stage_iter)
    {

        printf("------------------------------------------------------ Stage: %d ------------------------------------------------------\n", k);
        std::cout << "Parameters Settings (Penalty):  wP_lam = " << wP_lam << ", wP_kap = " << wP_kap << "\n";
        std::cout << "Parameters Settings (Regular):  wM_kap = " << wM_kap << ", wL_kap = " << wL_kap << ", wM_lam = " << wM_lam << ", wL_lam = " << wL_lam << "\n";

        printf("----------------------------  OptKap Start ----------------------------\n", k);

        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        // OptKap: self_reg = kappa_reg (dynamic), lambda_reg is the constant other_reg.
        auto logger_OptKap = [&](int i, const Eigen::VectorXd& x_iter,
                                 double spn, double dist, double self_reg_iter, double /*penalty_iter*/) {
            const auto _proj_st = computeProjStateFrom(reshape_x_to_V(x_iter));
            const double pd      = _proj_st.dist;
            const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(_proj_st.V);
            // Always log BOTH penalties (full kappa-side and lambda-side
            // distance to the 1D feasible set at the current theta).
            const double pen_kap = penalty_to_kapp.eval(kappa_pf_s.toVector());
            const double pen_lam = penalty_to_lamb.eval(lambda_pf_s.toVector());
            iter_log_ofs << k << ",OptKap," << i << ","
                         << spn << "," << dist << "," << pd << ","
                         << self_reg_iter << "," << lambda_reg << ","
                         << pen_kap << "," << pen_lam << ","
                         << wP_kap << "," << wP_lam << ","
                         << wM_kap << "," << wL_kap << "," << wM_lam << "," << wL_lam << "," << bd_rms << "," << int_rms << "\n";
        };
        Vr = sparse_gauss_newton_FixLam_OptKap_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, lambda_reg,
                                                       adjointFunc_OptKap, penalty_to_kapp, fixedIdx,
                                                       config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                                                       wM_kap, wL_kap, kappa_anchor, wP_kap,
                                                       E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                                                       distance, spn_energy, self_reg,
                                                       logger_OptKap);
        kappa_reg = self_reg; // sync for the next OptLam call
        const double dist_after_kap = distance;
        const double proj_after_kap = computeProjectedDistance();
        {
            const auto _proj_st_end = computeProjStateFrom(Vr);
            const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(_proj_st_end.V);
            const double pen_kap = penalty_to_kapp.eval(kappa_pf_s.toVector());
            const double pen_lam = penalty_to_lamb.eval(lambda_pf_s.toVector());
            iter_log_ofs << k << ",OptKap,-1," << spn_energy << "," << distance << "," << proj_after_kap << ","
                         << kappa_reg << "," << lambda_reg << ","
                         << pen_kap << "," << pen_lam << ","
                         << wP_kap << "," << wP_lam << ","
                         << wM_kap << "," << wL_kap << "," << wM_lam << "," << wL_lam << "," << bd_rms << "," << int_rms << "\n";
        }

        printStageStats();

        printf("----------------------------  OptKap Finish ----------------------------\n", k);

        printf("----------------------------  OptLam Start ----------------------------\n", k);
        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        // OptLam: self_reg = lambda_reg (dynamic), kappa_reg is the constant other_reg.
        auto logger_OptLam = [&](int i, const Eigen::VectorXd& x_iter,
                                 double spn, double dist, double self_reg_iter, double /*penalty_iter*/) {
            const auto _proj_st = computeProjStateFrom(reshape_x_to_V(x_iter));
            const double pd      = _proj_st.dist;
            const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(_proj_st.V);
            const double pen_kap = penalty_to_kapp.eval(kappa_pf_s.toVector());
            const double pen_lam = penalty_to_lamb.eval(lambda_pf_s.toVector());
            iter_log_ofs << k << ",OptLam," << i << ","
                         << spn << "," << dist << "," << pd << ","
                         << kappa_reg << "," << self_reg_iter << ","
                         << pen_kap << "," << pen_lam << ","
                         << wP_kap << "," << wP_lam << ","
                         << wM_kap << "," << wL_kap << "," << wM_lam << "," << wL_lam << "," << bd_rms << "," << int_rms << "\n";
        };
        Vr = sparse_gauss_newton_FixKap_OptLam_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, kappa_reg,
                                                       adjointFunc_OptLam, penalty_to_lamb, fixedIdx,
                                                       config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                                                       wM_lam, wL_lam, lambda_anchor, wP_lam,
                                                       E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                                                       distance, spn_energy, self_reg,
                                                       logger_OptLam);
        lambda_reg = self_reg; // sync for the next OptKap call
        const double dist_after_lam = distance;
        const double proj_after_lam = computeProjectedDistance();
        {
            const auto _proj_st_end = computeProjStateFrom(Vr);
            const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(_proj_st_end.V);
            const double pen_kap = penalty_to_kapp.eval(kappa_pf_s.toVector());
            const double pen_lam = penalty_to_lamb.eval(lambda_pf_s.toVector());
            iter_log_ofs << k << ",OptLam,-1," << spn_energy << "," << distance << "," << proj_after_lam << ","
                         << kappa_reg << "," << lambda_reg << ","
                         << pen_kap << "," << pen_lam << ","
                         << wP_kap << "," << wP_lam << ","
                         << wM_kap << "," << wL_kap << "," << wM_lam << "," << wL_lam << "," << bd_rms << "," << int_rms << "\n";
        }

        printStageStats();

        printf("----------------------------  OptLam Finish ----------------------------\n", k);

        // ---- Joint snap to feasible material BEFORE P-update --------------
        // Controlled by cfg.json -> RuntimeSettings.snap_before_P (default false).
        //
        // When ON: continuous (lambda, kappa) is hard-snapped to the nearest
        // feasible material pair before the P-update.  Each stage becomes a
        // proximal step (continuous SGN -> hard snap -> P-update), so P is
        // aligned with the actually-manufactured (discrete) design.
        // After OptP finish, `Distance` (continuous) equals `Projected
        // distance` (snap-then-forward).
        //
        // When OFF: P-update operates on the continuous SGN result.  Useful
        // for debugging / comparing the "soft homotopy + ARAP" flow against
        // the proximal flow.
        if (config.RuntimeSetting.snap_before_P)
        {
            for (Face f : mesh.faces())
            {
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb,
                                            kappa_pf_s[f], lambda_pf_s[f]);
                lambda_pf_s[f] = ac.feasible_lamb[idx];
                kappa_pf_s[f]  = ac.feasible_kapp[idx];
            }
        }

        // ---- OptP via SGN on continuous material --------------------------
        // The 5-stage BCD loop uses continuous (lambda, kappa) throughout.
        // After the loop exits + best snapshot is restored, a single extra
        // SGN OptP on SNAPPED material is run (outside this loop) to refine
        // P against the actually-manufactured discrete design.
        printf("----------------------------  OptP Start ------------------------------\n");
        {
            auto adjointFunc_OptP = adjointFunction_FixMaterial_OptP(
                geometry, F, lambda_pf_s, kappa_pf_s,
                E, nu, ac.thickness,
                config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);

            const double wM_P = config.RuntimeSetting.wM_P;
            const double wL_P = config.RuntimeSetting.wL_P;

            auto logger_OptP = [&](int i, const Eigen::VectorXd& x_iter,
                                   double spn, double dist, double self_reg_iter, double /*pen*/) {
                // OptP runs on continuous material here, so `dist` is the
                // continuous-material distance.  proj_dist needs an explicit
                // snap-then-forward-sim call to be the true projected
                // distance (consistent with OptKap / OptLam loggers).
                const auto _proj_st = computeProjStateFrom(reshape_x_to_V(x_iter));
                const double pd      = _proj_st.dist;
                const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(_proj_st.V);
                const double pen_kap = penalty_to_kapp.eval(kappa_pf_s.toVector());
                const double pen_lam = penalty_to_lamb.eval(lambda_pf_s.toVector());
                iter_log_ofs << k << ",OptP," << i << ","
                             << spn << "," << dist << "," << pd << ","
                             << kappa_reg << "," << lambda_reg << ","
                             << pen_kap << "," << pen_lam << ","
                             << wP_kap << "," << wP_lam << ","
                             << wM_kap << "," << wL_kap << "," << wM_lam << "," << wL_lam << "," << bd_rms << "," << int_rms << "\n";
            };

            double P_reg = 0.0;
            Vr = sparse_gauss_newton_FixMaterial_OptP(
                geometry, F, targetV, Vr, P,
                lambda_pf_s, kappa_pf_s,               // continuous material
                masses, M_P_2, L_P_2, P_anchor,
                kappa_reg + lambda_reg,                // other_reg
                adjointFunc_OptP, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                wM_P, wL_P,
                E, nu, ac.thickness,
                config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                distance, spn_energy, P_reg,
                logger_OptP);

            MrInv = precomputeMrInv(mesh, P, F);
            M_kappa = computeFaceMassKappa(mesh, MrInv);
            distance = recomputeForwardState();
            kappa_reg = computeKappaReg();
            lambda_reg = computeLambdaReg();
            spn_energy = distance + kappa_reg + lambda_reg;

            // Dump P for inspection.
            Eigen::MatrixXd P_obj(P.rows(), 3);
            P_obj.leftCols(2) = P;
            P_obj.col(2).setZero();
            igl::writeOBJ(morph_dir + "patch_0_P_stage" + std::to_string(k) + ".obj", P_obj, F);

            std::cout << "[OptP finish] stage " << k << "  ";
            printStageStats();
            const double pd_optp = computeProjectedDistance();
            const auto _proj_st_end = computeProjStateFrom(Vr);
            const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(_proj_st_end.V);
            const double pen_kap = penalty_to_kapp.eval(kappa_pf_s.toVector());
            const double pen_lam = penalty_to_lamb.eval(lambda_pf_s.toVector());
            iter_log_ofs << k << ",OptP,-1," << spn_energy << "," << distance << "," << pd_optp << ","
                         << kappa_reg << "," << lambda_reg << ","
                         << pen_kap << "," << pen_lam << ","
                         << wP_kap << "," << wP_lam << ","
                         << wM_kap << "," << wL_kap << "," << wM_lam << "," << wL_lam << "," << bd_rms << "," << int_rms << "\n";
        }
        // -------------------------------------------------------------------

        // ---- Trust-region safeguard: accept / reject this stage -----------
        // Trust-region safeguard:
        //   * best snapshot updates iff proj_new < proj_best, so
        //     `proj_best` strictly tracks historical minimum proj
        //   * reject (revert + shrink wP growth) only fires when BOTH dist
        //     and proj worsen (matches the user's original spec)
        //   * post-loop restore guarantees final outputs reflect the
        //     historical-best snapshot
        const double dist_new = distance;
        const double proj_new = computeProjectedDistance();

        const bool snapshot_improves = (proj_new < proj_best);
        if (snapshot_improves)
        {
            proj_best       = proj_new;
            dist_best       = dist_new;
            Vr_best         = Vr;
            lambda_pf_best  = lambda_pf_s;
            kappa_pf_best   = kappa_pf_s;
            P_best          = P;
            MrInv_best      = MrInv;
            M_kappa_best    = M_kappa;
            wP_lam_best     = wP_lam;
            wP_kap_best     = wP_kap;
            kappa_reg_best  = kappa_reg;
            lambda_reg_best = lambda_reg;
        }

        // Stage-level reject: both dist and proj worsened relative to the
        // current best snapshot.  Revert all state and halve the single wP
        // growth factor for the next attempt.
        const bool reject = (dist_new > dist_best) && (proj_new > proj_best);
        if (reject)
        {
            Vr          = Vr_best;
            lambda_pf_s = lambda_pf_best;
            kappa_pf_s  = kappa_pf_best;
            P           = P_best;
            MrInv       = MrInv_best;
            M_kappa     = M_kappa_best;
            // wM / wL are part of the homotopy schedule, NOT a state-level
            // quantity -- they should keep monotonically decaying even when
            // a stage is rejected.  Same logic argues against rolling back
            // kappa_reg / lambda_reg (recomputed below from snapshot theta).
            wP_lam      = wP_lam_best;
            wP_kap      = wP_kap_best;
            kappa_reg   = kappa_reg_best;
            lambda_reg  = lambda_reg_best;
            wP_lam_growth_factor = std::max(1e-4, wP_lam_growth_factor * 0.5);
            wP_kap_growth_factor = std::max(1e-4, wP_kap_growth_factor * 0.5);
            std::cout << "[REJECT] stage " << k
                      << ": dist=" << dist_new << ">" << dist_best
                      << " AND proj=" << proj_new << ">" << proj_best
                      << "; revert state, wP_lam_growth -> " << wP_lam_growth_factor
                      << ", wP_kap_growth -> " << wP_kap_growth_factor << "\n";
        }
        else if (snapshot_improves)
        {
            std::cout << "[ACCEPT, best updated] stage " << k
                      << ": dist=" << dist_new << "  proj=" << proj_new
                      << " (best now)\n";
        }
        else
        {
            std::cout << "[ACCEPT, best unchanged] stage " << k
                      << ": dist=" << dist_new << "  proj=" << proj_new
                      << " (best proj still " << proj_best << ")\n";
        }
        // -------------------------------------------------------------------

        k++;
        // Unconditional homotopy growth: safeguard (revert + shrink growth
        // factor on REJECT) controls overshoot, so the threshold check is
        // no longer needed.  wP_lam and wP_kap are independent and grow
        // each stage.
        wP_lam *= (1.0 + wP_lam_growth_factor);
        wP_kap *= (1.0 + wP_kap_growth_factor);

        // if (penalty_kap < penalty_threshold && penalty_lam < penalty_threshold)
        //     break;

        wM_kap *= 0.5;
        wL_kap *= 0.5;
        wM_lam *= 0.5;
        wL_lam *= 0.5;

        // Recompute reg accumulators after weight decay so the next stage's
        // SPN energy formula uses the *new* weights consistently for both
        // kappa_reg and lambda_reg.  Also picks up the refreshed M_kappa
        // from the P-update above.
        kappa_reg = computeKappaReg();
        lambda_reg = computeLambdaReg();

        printf("----------------------------------------------------------------------------------------------------------------------\n");
    }

    // Stage loop exited.  Restore the best snapshot explicitly so the
    // downstream material projection / final proj.obj is guaranteed to be
    // the historical-minimum-projected-distance design — independent of
    // whether the last stage's snapshot got updated.
    Vr          = Vr_best;
    lambda_pf_s = lambda_pf_best;
    kappa_pf_s  = kappa_pf_best;
    P           = P_best;
    MrInv       = MrInv_best;
    M_kappa     = M_kappa_best;
    std::cout << "Restored best snapshot: dist=" << dist_best
              << "  proj=" << proj_best << "\n";

    // ---- Final snap-material OptP -----------------------------------------
    // The BCD loop optimised everything in continuous space.  One extra SGN
    // OptP on the snapped (= actually manufacturable) material now refines
    // P so the manufactured forward-sim lands as close to V_T as possible.
    printf("----------------------------  Final SNAP OptP ------------------------------\n");
    {
        FaceData<double> lambda_pf_snap(mesh);
        FaceData<double> kappa_pf_snap(mesh);
        for (Face f : mesh.faces()) {
            int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb,
                                        kappa_pf_s[f], lambda_pf_s[f]);
            lambda_pf_snap[f] = ac.feasible_lamb[idx];
            kappa_pf_snap[f]  = ac.feasible_kapp[idx];
        }
        auto adjointFunc_OptP_snap = adjointFunction_FixMaterial_OptP(
            geometry, F, lambda_pf_snap, kappa_pf_snap,
            E, nu, ac.thickness,
            config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);

        auto logger_OptP_snap = [&](int i, const Eigen::VectorXd& x_iter,
                                    double spn, double dist, double, double) {
            // FinalSnapOptP runs after the stage loop on snapped material:
            // dist itself IS the projected distance (snap-equilibrium V_r
            // vs V_T).  Still log the current values of penalty / weights
            // so the CSV row never contains placeholder zeros.
            const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(reshape_x_to_V(x_iter));
            const double pen_kap = penalty_to_kapp.eval(kappa_pf_s.toVector());
            const double pen_lam = penalty_to_lamb.eval(lambda_pf_s.toVector());
            iter_log_ofs << "-1,FinalSnapOptP," << i << ","
                         << spn << "," << dist << "," << dist << ","
                         << kappa_reg << "," << lambda_reg << ","
                         << pen_kap << "," << pen_lam << ","
                         << wP_kap << "," << wP_lam << ","
                         << wM_kap << "," << wL_kap << "," << wM_lam << "," << wL_lam << "," << bd_rms << "," << int_rms << "\n";
        };

        double dummy_dist = 0, dummy_spn = 0, dummy_reg = 0;
        Vr = sparse_gauss_newton_FixMaterial_OptP(
            geometry, F, targetV, Vr, P,
            lambda_pf_snap, kappa_pf_snap,
            masses, M_P_2, L_P_2, P_anchor,
            0.0,                                   // other_reg = 0 for final snap pass
            adjointFunc_OptP_snap, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
            config.RuntimeSetting.wM_P, config.RuntimeSetting.wL_P,
            E, nu, ac.thickness,
            config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
            dummy_dist, dummy_spn, dummy_reg,
            logger_OptP_snap);
        // Refresh MrInv with the final P; (lambda_pf_s, kappa_pf_s) stay
        // continuous - the snap is only used inside the OptP.
        MrInv = precomputeMrInv(mesh, P, F);
        std::cout << "[Final SNAP OptP done] dist=" << dummy_dist << "\n";
    }

    // V_target was already in physical (device) units; Vr lives in the same
    // frame, so write it out as-is — no inverse rescaling.
    igl::writeOBJ(morph_dir + "patch_0_inv.obj", Vr, F);

    // ---- Final projected forward sim (manufacturing reality) ----------------
    // Snap (lambda, kappa) per face to the nearest feasible (t1, t2) pair,
    // run forward Newton on the snapped material, then write out the
    // resulting mesh as patch_0_proj.obj — this is what the device will
    // actually produce.  Also report the final manufacturing distance
    // prominently in the log.
    double final_proj_dist = 0.0;
    double final_bd_rms = 0.0;
    double final_int_rms = 0.0;
    {
        FaceData<double> kappa_pf_proj(mesh);
        FaceData<double> lambda_pf_proj(mesh);
        for (Face f : mesh.faces())
        {
            int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb,
                                        kappa_pf_s[f], lambda_pf_s[f]);
            kappa_pf_proj[f]  = ac.feasible_kapp[idx];
            lambda_pf_proj[f] = ac.feasible_lamb[idx];
        }
        auto simFunc_proj = simulationFunction(geometry, MrInv, lambda_pf_proj, kappa_pf_proj,
                                               E, nu, ac.thickness,
                                               config.RuntimeSetting.w_s,
                                               config.RuntimeSetting.w_b, ref_faces);
        Eigen::MatrixXd Vr_proj = Vr;
        newton(geometry, Vr_proj, simFunc_proj,
               config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx);

        for (size_t i = 0; i < nV; ++i)
            for (int j = 0; j < 3; ++j)
            {
                double d = Vr_proj(i, j) - targetV(i, j);
                final_proj_dist += masses(3 * i + j) * d * d;
            }

        // bd / int RMS computed on the actually-manufactured proj mesh.
        auto [_bd, _int] = computeBoundaryInteriorRMS(Vr_proj);
        final_bd_rms  = _bd;
        final_int_rms = _int;

        const std::string proj_path = morph_dir + "patch_0_proj.obj";
        igl::writeOBJ(proj_path, Vr_proj, F);
        spdlog::info("Proj mesh -> {}", proj_path);
    }

    // ---- Material projection: per-face (lambda, kappa) -> nearest feasible (t1, t2) ----
    // Writes two files:
    //   patch_0_material.txt  : face_id  t1  t2           (discrete grayscale doses)
    //   patch_0_lamkap.txt    : face_id  lambda  kappa    (continuous SGN values)
    {
        const std::string mat_path = design_dir + "patch_0_material.txt";
        const std::string lk_path = design_dir + "patch_0_lamkap.txt";
        std::ofstream mof(mat_path);
        std::ofstream lof(lk_path);
        mof << "# face_id  t1  t2\n";
        lof << "# face_id  lambda  kappa\n";
        for (Face f : mesh.faces())
        {
            double kap = kappa_pf_s[f];
            double lam = lambda_pf_s[f];
            int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap, lam);
            double t1 = ac.feasible_t_vals[idx].first;
            double t2 = ac.feasible_t_vals[idx].second;
            mof << f.getIndex() << "  " << t1 << "  " << t2 << "\n";
            lof << f.getIndex() << "  " << lam << "  " << kap << "\n";
        }
        spdlog::info("Material -> {}", mat_path);
        spdlog::info("LamKap   -> {}", lk_path);
    }

    // ---- Highlight: final manufacturing distance ----
    std::cout << "\n";
    std::cout << "==========================================================\n";
    std::cout << "  FINAL Projected distance (manufactured design):  "
              << final_proj_dist << "\n";
    std::cout << "  Per-vertex RMS (Euclidean, mm)  bd = " << final_bd_rms
              << "   int = " << final_int_rms << "\n";
    std::cout << "==========================================================\n";
    std::cout << "\n";

    spdlog::info("Inverse (single mesh): done.  Output -> {}", morph_dir);
    return 0;
}
