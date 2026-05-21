

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
// #include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/loop.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <filesystem>
#include <io.h>

#include <spdlog/spdlog.h>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include"config.hpp"
#include "material.hpp"
#include "parameterization.h"
#include "simulation_utils.h"
#include "functions.h"
#include "newton.h"
#include "LocalGlobalSolver.h"
#include "morphmesh.hpp"
#include "morph_functions.hpp"
#include "output.hpp"
#include "boundary_utils.h"

// #define __VERFIY_FORWARD_PREDIT__
#define __VERFIY_INVERSE_DESIGN__

#define __Add_PENALTY__

int main(int argc, char* argv[])
{


    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    Config config("cfg.json");
    ActiveComposite ac(config.materialJsonPath());
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    spdlog::info("program start (multi-patch inverse design):");

    ///***************************************** Patch I/O setup *****************************************///

    const std::string model       = config.ModelSetting.ModelName;
    const std::string target_dir  = config.PathSetting.TargetDir + model + "/";
    const std::string param_dir   = config.PathSetting.ParamDir  + model + "/";
    const std::string cond_dir    = config.PathSetting.CondDir   + model + "/";
    const std::string design_dir  = config.PathSetting.DesignDir + model + "/";
    const std::string morph_dir   = config.PathSetting.MorphDir  + model + "/";
    std::filesystem::create_directories(morph_dir);
    std::filesystem::create_directories(design_dir);
    spdlog::info("Target dir : {}", target_dir);
    spdlog::info("Param  dir : {}", param_dir);
    spdlog::info("Cond   dir : {}", cond_dir);
    spdlog::info("Design dir : {}", design_dir);
    spdlog::info("Morph  dir : {}", morph_dir);

    ///***************************************** Phase 1: read pre-scaled V/P from disk *****************************************///
    // V comes from TargetDir (scaled), P comes from ParamDir (scaled).  ParamAll
    // must have produced these + global_scale.txt + per-patch cond files.

    struct PatchData {
        size_t idx;
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        Eigen::MatrixXd P;
        size_t nV, nF;
    };
    std::vector<PatchData> patches;

    for (size_t i = 0;; ++i) {
        std::string v_path = target_dir + "patch_" + std::to_string(i) + "_V.obj";
        std::string p_path = param_dir  + "patch_" + std::to_string(i) + "_P.obj";
        Eigen::MatrixXd Vp;
        Eigen::MatrixXi Fp;
        if (!igl::readOBJ(v_path, Vp, Fp)) {
            spdlog::info("No more patches after idx {}.", i - 1);
            break;
        }
        Eigen::MatrixXd P_loaded3;
        Eigen::MatrixXi F_p;
        if (!igl::readOBJ(p_path, P_loaded3, F_p)) {
            spdlog::error("Patch {} V exists but P missing: {}.  Run ParamAll first.", i, p_path);
            return -1;
        }
        if (P_loaded3.rows() != Vp.rows()) {
            spdlog::error("Patch {}: P vertex count ({}) != V count ({}).",
                          i, P_loaded3.rows(), Vp.rows());
            return -1;
        }

        PatchData pd;
        pd.idx = i;
        pd.V = std::move(Vp);
        pd.F = std::move(Fp);
        pd.P = P_loaded3.leftCols(2);
        pd.nV = pd.V.rows();
        pd.nF = pd.F.rows();
        spdlog::info("Patch {}: read {} V, {} F.", i, pd.nV, pd.nF);
        patches.push_back(std::move(pd));
    }
    spdlog::info("Total patches: {}", patches.size());

    if (patches.empty()) {
        spdlog::error("No patches found, abort.");
        return -1;
    }

    ///***************************************** Phase 2: per-patch inverse design *****************************************///
    // V/P are already in physical (device) units; no re-scaling here.

    for (auto& pd : patches) {
        spdlog::info("=================================================");
        spdlog::info("Inverse design for patch {} ({} V, {} F)", pd.idx, pd.nV, pd.nF);
        spdlog::info("=================================================");

        // V and P are already scaled by globalScale (done by ParamAll).
        // Aliases so the existing inverse-design code below reads naturally.
        Eigen::MatrixXd& V = pd.V;
        Eigen::MatrixXi& F = pd.F;
        Eigen::MatrixXd& P = pd.P;
        size_t nV = pd.nV;
        size_t nF = pd.nF;

        ManifoldSurfaceMesh mesh(F);
        VertexPositionGeometry geometry(mesh, V);
        geometry.refreshQuantities();

        // Boundary-face reference mapping (BFS on dual graph).
        std::vector<bool> is_boundary_face;
        std::vector<int> ref_faces = buildRefFaces(mesh, is_boundary_face);

        ///***************************************** Material Settings *****************************************///

        spdlog::info("Step 2: Material Settings.");

        Eigen::MatrixXd targetV = V;

        FaceData<Eigen::MatrixXd> M = precomputeM(mesh, V, F);
        FaceData<Eigen::Matrix2d> MrInv = precomputeMrInv(mesh, P, F);

        // Boundary condition: read 3 vertex indices from cond file written by ParamAll
        std::vector<int> fixedVertexIdx;
        {
            const std::string cond_path = cond_dir + "patch_" + std::to_string(pd.idx) + "_bound_center.txt";
            std::ifstream ifs(cond_path);
            if (!ifs.is_open()) {
                spdlog::error("Cannot read cond file: {}.  Run ParamAll first.", cond_path);
                return -1;
            }
            int v0, v1, v2;
            ifs >> v0 >> v1 >> v2;
            fixedVertexIdx = {v0, v1, v2};
            spdlog::info("Patch {} cond: v0={} v1={} v2={}", pd.idx, v0, v1, v2);
        }
        std::vector<int> fixedIdx;  // 9 DOF indices
        for (int v : fixedVertexIdx)
            for (int k = 0; k < 3; ++k) fixedIdx.push_back(3 * v + k);
        std::sort(fixedIdx.begin(), fixedIdx.end());

        double E = 1.0;
        double nu = 0.5;
        Morphmesh morph_mesh(V, P, F, E, nu);
        Morphmesh::ComputeMorphophing(geometry, V, F, nV, nF, ref_faces,
            MrInv, morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t, morph_mesh.kappa_pv_t, morph_mesh.kappa_pf_t, &morph_mesh.vertex_area_sum);
        Morphmesh::SetMorphophing(morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t,
            morph_mesh.kappa_pv_t,morph_mesh.kappa_pf_t,
            morph_mesh.lambda_pv_s, morph_mesh.lambda_pf_s,
            morph_mesh.kappa_pf_s, morph_mesh.kappa_pf_s);

        // Mirror physical-unit target into morph_dir so {targ, inv} sit side-by-side.
        igl::writeOBJ(morph_dir + "patch_" + std::to_string(pd.idx) + "_targ.obj", V, F);


        VertexData<double> lambda_pv_s(mesh, morph_mesh.lambda_pv_s);
        FaceData<double> lambda_pf_s(mesh, morph_mesh.lambda_pf_s);
        FaceData<double> kappa_pf_s(mesh, morph_mesh.kappa_pf_s);

        ///***************************************** Inverse Design *****************************************///

        auto V_pred = V;
        // V_init: flat plate aligned so fixed vertices match V exactly.
        Eigen::MatrixXd Vr = flatPlateAligned(P, V, fixedVertexIdx);
        igl::writeOBJ(morph_dir + "patch_" + std::to_string(pd.idx) + "_init.obj", Vr, F);

        // Lumped vertex mass vector (size 3*nV) -- shared by all SGN calls so
        // distance/SPN values match between main and the solver exactly.
        const Eigen::VectorXd masses = computeVertexMasses(geometry);

        // Face-space matrices to compute the other-variable regulariser as
        // a constant offset, so OptKap/OptLam stages share a unified SPN
        // energy formula (distance + kappa_reg + lambda_reg).
        // M_kappa depends on MrInv -> refreshed on every P-update.
        Eigen::SparseMatrix<double> M_kappa = computeFaceMassKappa(mesh, MrInv);
        const Eigen::SparseMatrix<double> M_lambda = computeFaceMassLambda(geometry);
        const Eigen::SparseMatrix<double> L_face = computeFaceDualLaplacian(mesh);

        // ARAP solver for the per-stage lambda-aware P-update.  Built once
        // per patch from (V, F); cotmatrix factorisation reused.
        LocalGlobalSolver paramSolver(V, F);

        spdlog::info("Step 4: Inverse Design.");

        double wP_kap = config.RuntimeSetting.wP_kap;
        double wP_lam = config.RuntimeSetting.wP_lam;
        double penalty_threshold = config.RuntimeSetting.penalty_threshold;
        double betaP = config.RuntimeSetting.betaP;
        auto penalty_to_lamb = MaterialPenaltyFunctionPerF(geometry, ac.feasible_lamb, betaP);
        auto penalty_to_kapp = MaterialPenaltyFunctionPerF(geometry, ac.feasible_kapp, betaP);
        auto penalty_to_modu = MaterialPenaltyFunctionPerV(geometry, ac.feasible_modl, betaP);

        int stage_iter = config.RuntimeSetting.stage_iter;
        int k = 0;

        double wM_kap = config.RuntimeSetting.wM_kap;
        double wM_lam = config.RuntimeSetting.wM_lam;
        double wL_kap = config.RuntimeSetting.wL_kap;
        double wL_lam = config.RuntimeSetting.wL_lam;

#ifdef __Add_PENALTY__

        double distance = 0.0;
        double spn_energy = 0.0;
        double penalty_kap = 0.0;
        double penalty_lam = 0.0;

        auto computeKappaReg = [&]() {
            const Eigen::VectorXd k = kappa_pf_s.toVector();
            return wM_kap * k.dot(M_kappa * k) + wL_kap * k.dot(L_face * k);
        };
        auto computeLambdaReg = [&]() {
            const Eigen::VectorXd l = lambda_pf_s.toVector();
            return wM_lam * l.dot(M_lambda * l) + wL_lam * l.dot(L_face * l);
        };
        double kappa_reg  = computeKappaReg();
        double lambda_reg = computeLambdaReg();
        double self_reg   = 0.0;

        // Projected distance: snap (kappa, lambda) to nearest feasible per face,
        // re-run forward Newton, return mass-weighted distance to target.
        auto computeProjectedDistance = [&]() -> double {
            FaceData<double> kappa_pf_proj(mesh);
            FaceData<double> lambda_pf_proj(mesh);
            for (Face f : mesh.faces()) {
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb,
                                            kappa_pf_s[f], lambda_pf_s[f]);
                kappa_pf_proj[f]  = ac.feasible_kapp[idx];
                lambda_pf_proj[f] = ac.feasible_lamb[idx];
            }
            auto simFunc_proj = simulationFunction(geometry, MrInv, lambda_pf_proj, kappa_pf_proj,
                E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Eigen::MatrixXd Vr_proj = Vr;
            newton(geometry, Vr_proj, simFunc_proj,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx);
            double d2 = 0.0;
            for (size_t i = 0; i < nV; ++i)
                for (int j = 0; j < 3; ++j) {
                    double d = Vr_proj(i, j) - targetV(i, j);
                    d2 += masses(3 * i + j) * d * d;
                }
            return d2;
        };

        // Re-run forward Newton on current (P, lambda, kappa) to bring Vr back
        // to equilibrium after a P-update changed MrInv.
        auto recomputeForwardState = [&]() -> double {
            auto simFunc = simulationFunction(geometry, MrInv, lambda_pf_s, kappa_pf_s,
                E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            newton(geometry, Vr, simFunc,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx);
            double d2 = 0.0;
            for (size_t i = 0; i < nV; ++i)
                for (int j = 0; j < 3; ++j) {
                    double d = Vr(i, j) - targetV(i, j);
                    d2 += masses(3 * i + j) * d * d;
                }
            return d2;
        };

        // ---- Trust-region best snapshot ------------------------------------
        double dist_best = std::numeric_limits<double>::infinity();
        double proj_best = std::numeric_limits<double>::infinity();
        Eigen::MatrixXd            Vr_best         = Vr;
        FaceData<double>           lambda_pf_best  = lambda_pf_s;
        FaceData<double>           kappa_pf_best   = kappa_pf_s;
        Eigen::MatrixXd            P_best          = P;
        FaceData<Eigen::Matrix2d>  MrInv_best      = MrInv;
        Eigen::SparseMatrix<double> M_kappa_best   = M_kappa;
        double wM_kap_best = wM_kap, wL_kap_best = wL_kap;
        double wM_lam_best = wM_lam, wL_lam_best = wL_lam;
        double wP_kap_best = wP_kap, wP_lam_best = wP_lam;
        double kappa_reg_best  = kappa_reg;
        double lambda_reg_best = lambda_reg;
        double wP_growth_factor_kap = config.RuntimeSetting.wP_growth_factor_kap;
        double wP_growth_factor_lam = config.RuntimeSetting.wP_growth_factor_lam;

        // Per-patch CSV log: written under MorphLogsDir/{method}/{model}/.
        // Schema: stage,substage,iter,spn,dist.
        const std::string morphlogs_dir = config.PathSetting.MorphLogsDir
                                        + config.RuntimeSetting.morph_method + "/"
                                        + model + "/";
        std::filesystem::create_directories(morphlogs_dir);
        std::ofstream iter_log_ofs(morphlogs_dir + "patch_" + std::to_string(pd.idx) + "_iter_log.csv");
        iter_log_ofs << "stage,substage,iter,spn,dist\n";
        spdlog::info("Patch {} iter log -> {}", pd.idx,
                     morphlogs_dir + "patch_" + std::to_string(pd.idx) + "_iter_log.csv");

        while (k < stage_iter)
        {
            spdlog::info("Patch {} Stage {}: wP_kap={:.6f}, wP_lam={:.6f}, wM_kap={:.6f}, wL_kap={:.6f}, wM_lam={:.6f}, wL_lam={:.6f}",
                         pd.idx, k, wP_kap, wP_lam, wM_kap, wL_kap, wM_lam, wL_lam);

            // -- OptKap --
            auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            auto logger_OptKap = [&iter_log_ofs, k](int i, const Eigen::VectorXd&,
                                                     double spn, double dist,
                                                     double /*self_reg*/, double /*pen*/) {
                iter_log_ofs << k << ",OptKap," << i << "," << spn << "," << dist << "\n";
            };
            Vr = sparse_gauss_newton_FixLam_OptKap_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, lambda_reg,
                adjointFunc_OptKap, penalty_to_kapp, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_kap, wL_kap, wP_kap,
                E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                distance, spn_energy, self_reg,
                logger_OptKap);
            kappa_reg = self_reg;
            iter_log_ofs << k << ",OptKap,-1," << spn_energy << "," << distance << "\n";
            penalty_kap = compute_candidate_diff(ac.feasible_kapp, kappa_pf_s.toVector(), true);
            penalty_lam = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
            spdlog::info("Patch {} Stage {} [OptKap finish] SPN={:.6f} Dist={:.6f} ProjDist={:.6f} Pkap={:.6f} Plam={:.6f}",
                         pd.idx, k, spn_energy, distance, computeProjectedDistance(), penalty_kap, penalty_lam);

            // -- OptLam --
            auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            auto logger_OptLam = [&iter_log_ofs, k](int i, const Eigen::VectorXd&,
                                                     double spn, double dist,
                                                     double /*self_reg*/, double /*pen*/) {
                iter_log_ofs << k << ",OptLam," << i << "," << spn << "," << dist << "\n";
            };
            Vr = sparse_gauss_newton_FixKap_OptLam_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, kappa_reg,
                adjointFunc_OptLam, penalty_to_lamb, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_lam, wL_lam, wP_lam,
                E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                distance, spn_energy, self_reg,
                logger_OptLam);
            lambda_reg = self_reg;
            iter_log_ofs << k << ",OptLam,-1," << spn_energy << "," << distance << "\n";
            penalty_kap = compute_candidate_diff(ac.feasible_kapp, kappa_pf_s.toVector(), true);
            penalty_lam = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
            spdlog::info("Patch {} Stage {} [OptLam finish] SPN={:.6f} Dist={:.6f} ProjDist={:.6f} Pkap={:.6f} Plam={:.6f}",
                         pd.idx, k, spn_energy, distance, computeProjectedDistance(), penalty_kap, penalty_lam);

            // -- Optional joint snap (lambda, kappa) before P-update --
            if (config.RuntimeSetting.snap_before_P) {
                for (Face f : mesh.faces()) {
                    int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb,
                                                kappa_pf_s[f], lambda_pf_s[f]);
                    lambda_pf_s[f] = ac.feasible_lamb[idx];
                    kappa_pf_s[f]  = ac.feasible_kapp[idx];
                }
            }

            // -- Lambda-aware ARAP P-update --
            {
                Eigen::VectorXd lambdaVec = lambda_pf_s.toVector();
                // Clamp lambda to a safe positive range before inverting so a
                // pathological zero/negative lambda from SGN cannot push
                // sTarget to inf/nan and trash the sparse factorisation.
                for (int ii = 0; ii < lambdaVec.size(); ++ii) {
                    if (!std::isfinite(lambdaVec(ii)) || lambdaVec(ii) < 1e-3)
                        lambdaVec(ii) = 1e-3;
                }
                Eigen::VectorXd sTarget = 1.0 / lambdaVec.array();
                Eigen::MatrixX2d P_2d = P;
                paramSolver.solve(P_2d, sTarget, sTarget, 10);
                P = P_2d;

                // Refresh P-dependent quantities.
                MrInv = precomputeMrInv(mesh, P, F);
                M_kappa = computeFaceMassKappa(mesh, MrInv);

                distance   = recomputeForwardState();
                kappa_reg  = computeKappaReg();
                lambda_reg = computeLambdaReg();
                spn_energy = distance + kappa_reg + lambda_reg;
                spdlog::info("Patch {} Stage {} [OptP   finish] SPN={:.6f} Dist={:.6f} ProjDist={:.6f} (lambda range [{:.4f},{:.4f}])",
                             pd.idx, k, spn_energy, distance, computeProjectedDistance(),
                             lambdaVec.minCoeff(), lambdaVec.maxCoeff());
                iter_log_ofs << k << ",OptP,-1," << spn_energy << "," << distance << "\n";
            }

            // -- Trust-region safeguard --
            //
            // Snapshot update is decoupled from accept/reject:
            //   * best snapshot updates iff proj_new < proj_best, so
            //     `proj_best` strictly tracks the historical minimum.
            //   * reject (revert all state + shrink wP growth) still uses
            //     the "both worsen" rule the user originally specified.
            //
            // Otherwise (dist improves but proj worsens) we ACCEPT the
            // step and let SGN keep advancing, but the best snapshot
            // intentionally does *not* drift to a worse-proj state.  This
            // way the post-loop restore guarantees the final manufacturing
            // distance equals the historical minimum proj.
            const double dist_new = distance;
            const double proj_new = computeProjectedDistance();

            const bool snapshot_improves = (proj_new < proj_best);
            if (snapshot_improves) {
                proj_best       = proj_new;
                dist_best       = dist_new;
                Vr_best         = Vr;
                lambda_pf_best  = lambda_pf_s;
                kappa_pf_best   = kappa_pf_s;
                P_best          = P;
                MrInv_best      = MrInv;
                M_kappa_best    = M_kappa;
                wM_kap_best     = wM_kap;   wL_kap_best = wL_kap;
                wM_lam_best     = wM_lam;   wL_lam_best = wL_lam;
                wP_kap_best     = wP_kap;   wP_lam_best = wP_lam;
                kappa_reg_best  = kappa_reg;
                lambda_reg_best = lambda_reg;
            }

            const bool reject = (dist_new > dist_best) && (proj_new > proj_best);
            if (reject) {
                Vr          = Vr_best;
                lambda_pf_s = lambda_pf_best;
                kappa_pf_s  = kappa_pf_best;
                P           = P_best;
                MrInv       = MrInv_best;
                M_kappa     = M_kappa_best;
                wM_kap      = wM_kap_best;  wL_kap = wL_kap_best;
                wM_lam      = wM_lam_best;  wL_lam = wL_lam_best;
                wP_kap      = wP_kap_best;  wP_lam = wP_lam_best;
                kappa_reg   = kappa_reg_best;
                lambda_reg  = lambda_reg_best;
                wP_growth_factor_kap = std::max(1e-4, wP_growth_factor_kap * 0.5);
                wP_growth_factor_lam = std::max(1e-4, wP_growth_factor_lam * 0.5);
                spdlog::info("Patch {} Stage {} [REJECT] dist={:.6f}>{:.6f} AND proj={:.6f}>{:.6f}; revert, wP_growth_factor_kap -> {:.6f}, _lam -> {:.6f}",
                             pd.idx, k, dist_new, dist_best, proj_new, proj_best,
                             wP_growth_factor_kap, wP_growth_factor_lam);
            } else if (snapshot_improves) {
                spdlog::info("Patch {} Stage {} [ACCEPT, best updated] dist={:.6f} proj={:.6f} (best now)",
                             pd.idx, k, dist_new, proj_new);
            } else {
                spdlog::info("Patch {} Stage {} [ACCEPT, best unchanged] dist={:.6f} proj={:.6f} (best proj still {:.6f})",
                             pd.idx, k, dist_new, proj_new, proj_best);
            }

            k++;
            if (penalty_kap >= penalty_threshold) wP_kap *= (1.0 + wP_growth_factor_kap);
            if (penalty_lam >= penalty_threshold) wP_lam *= (1.0 + wP_growth_factor_lam);

            wM_kap *= 0.5;  wL_kap *= 0.5;
            wM_lam *= 0.5;  wL_lam *= 0.5;
            kappa_reg  = computeKappaReg();
            lambda_reg = computeLambdaReg();
        }

        // After the stage loop, restore the best snapshot so the downstream
        // material output / proj.obj is guaranteed to reflect the historical
        // minimum projected distance, not the last (possibly worse) state.
        Vr          = Vr_best;
        lambda_pf_s = lambda_pf_best;
        kappa_pf_s  = kappa_pf_best;
        P           = P_best;
        MrInv       = MrInv_best;
        M_kappa     = M_kappa_best;
        spdlog::info("Patch {} restored best snapshot: dist={:.6f}  proj={:.6f}",
                     pd.idx, dist_best, proj_best);


#else

        Vr = targetV;
        double distance_kap = 0.0, spn_kap = 0.0, self_reg_kap = 0.0;
        double distance_lam = 0.0, spn_lam = 0.0, self_reg_lam = 0.0;
        double kappa_reg_np  = computeKappaReg();
        double lambda_reg_np = computeLambdaReg();
        while(k < stage_iter)
        {
            spdlog::info("Patch {} Stage {}, OptKap start", pd.idx, k);

            auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixLam_OptKap(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, lambda_reg_np,
                adjointFunc_OptKap, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, config.RuntimeSetting.wM_kap, config.RuntimeSetting.wL_kap,
                E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces,
                distance_kap, spn_kap, self_reg_kap);
            kappa_reg_np = self_reg_kap;

            spdlog::info("Patch {} Stage {}, OptKap finish - Distance: {:.6f}, SPN energy: {:.6f}", pd.idx, k, distance_kap, spn_kap);


            spdlog::info("Patch {} Stage {}, OptLam start", pd.idx, k);
            auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixKap_OptLam(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, kappa_reg_np,
                adjointFunc_OptLam, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, config.RuntimeSetting.wM_lam, config.RuntimeSetting.wL_lam,
                E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces,
                distance_lam, spn_lam, self_reg_lam);
            lambda_reg_np = self_reg_lam;

            spdlog::info("Patch {} Stage {}, OptLam finish - Distance: {:.6f}, SPN energy: {:.6f}", pd.idx, k, distance_lam, spn_lam);


            k++;
        }


#endif



        // Vr lives in the same physical frame as V; write as-is.
        igl::writeOBJ(morph_dir + "patch_" + std::to_string(pd.idx) + "_inv.obj", Vr, F);

        // ---- Final projected forward sim (manufacturing reality) ----------
        // Snap (lambda, kappa) per face to the nearest feasible (t1, t2) pair,
        // run forward Newton on the snapped material, write the resulting mesh
        // and report the manufacturing distance prominently.
        double final_proj_dist = 0.0;
        {
            FaceData<double> kappa_pf_proj(mesh);
            FaceData<double> lambda_pf_proj(mesh);
            for (Face f : mesh.faces()) {
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb,
                                            kappa_pf_s[f], lambda_pf_s[f]);
                kappa_pf_proj[f]  = ac.feasible_kapp[idx];
                lambda_pf_proj[f] = ac.feasible_lamb[idx];
            }
            auto simFunc_proj = simulationFunction(geometry, MrInv, lambda_pf_proj, kappa_pf_proj,
                E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Eigen::MatrixXd Vr_proj = Vr;
            newton(geometry, Vr_proj, simFunc_proj,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx);
            for (size_t i = 0; i < nV; ++i)
                for (int j = 0; j < 3; ++j) {
                    double d = Vr_proj(i, j) - targetV(i, j);
                    final_proj_dist += masses(3 * i + j) * d * d;
                }
            const std::string proj_path = morph_dir + "patch_" + std::to_string(pd.idx) + "_proj.obj";
            igl::writeOBJ(proj_path, Vr_proj, F);
            spdlog::info("Patch {} proj mesh -> {}", pd.idx, proj_path);
        }

        // ---- Material projection: per-face (lambda, kappa) -> nearest feasible (t1, t2) ----
        // Writes two files:
        //   patch_{i}_material.txt : face_id  t1  t2          (discrete grayscale)
        //   patch_{i}_lamkap.txt   : face_id  lambda  kappa   (continuous SGN values)
        {
            const std::string mat_path = design_dir + "patch_" + std::to_string(pd.idx) + "_material.txt";
            const std::string lk_path  = design_dir + "patch_" + std::to_string(pd.idx) + "_lamkap.txt";
            std::ofstream mof(mat_path);
            std::ofstream lof(lk_path);
            mof << "# face_id  t1  t2\n";
            lof << "# face_id  lambda  kappa\n";
            for (Face f : mesh.faces()) {
                double kap = kappa_pf_s[f];
                double lam = lambda_pf_s[f];
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap, lam);
                double t1 = ac.feasible_t_vals[idx].first;
                double t2 = ac.feasible_t_vals[idx].second;
                mof << f.getIndex() << "  " << t1  << "  " << t2  << "\n";
                lof << f.getIndex() << "  " << lam << "  " << kap << "\n";
            }
            spdlog::info("Patch {} material -> {}", pd.idx, mat_path);
            spdlog::info("Patch {} lamkap   -> {}", pd.idx, lk_path);
        }

        // Highlight: final manufacturing distance for this patch.
        std::cout << "\n";
        std::cout << "==========================================================\n";
        std::cout << "  Patch " << pd.idx
                  << "  FINAL Projected distance (manufactured): "
                  << final_proj_dist << "\n";
        std::cout << "==========================================================\n";
        std::cout << "\n";

        spdlog::info("Patch {} done.", pd.idx);
    }

    spdlog::info("All patches processed; program finish.");

}

