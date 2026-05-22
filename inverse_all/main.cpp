// InverseAll: inverse design over ALL patches of a segmented model.
// Reads pre-scaled V/P emitted by ParamAll; produces per-patch target / inv /
// material / lamkap outputs.  Per-patch inner loop mirrors `inverse/` --
// warm-up pure-SPN stages + MGDA stages + lambda-aware ARAP P-update.
//
// Reads (per patch i):
//   PathSetting.TargetDir + {model}/patch_i_V.obj
//   PathSetting.ParamDir  + {model}/patch_i_P.obj
//   PathSetting.CondDir   + {model}/patch_i_bound_center.txt
//
// Writes (per patch i):
//   PathSetting.MorphDir  + {model}/patch_i_targ.obj
//   PathSetting.MorphDir  + {model}/patch_i_init.obj
//   PathSetting.MorphDir  + {model}/patch_i_inv.obj
//   PathSetting.MorphDir  + {model}/patch_i_P_warmupN.obj   (P snapshot per ARAP update)
//   PathSetting.MorphDir  + {model}/patch_i_P_stageN.obj
//   PathSetting.DesignDir + {model}/patch_i_material.txt
//   PathSetting.DesignDir + {model}/patch_i_lamkap.txt

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/loop.h>
#include <igl/cotmatrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <limits>

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

int main(int /*argc*/, char* /*argv*/[])
{
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    Config config("cfg.json");
    ActiveComposite ac(config.materialJsonPath());
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    spdlog::info("InverseAll: start (multi-patch inverse design, MGDA).");

    const std::string model      = config.ModelSetting.ModelName;
    const std::string target_dir = config.PathSetting.TargetDir + model + "/";
    const std::string param_dir  = config.PathSetting.ParamDir  + model + "/";
    const std::string cond_dir   = config.PathSetting.CondDir   + model + "/";
    const std::string design_dir = config.PathSetting.DesignDir + model + "/";
    const std::string morph_dir  = config.PathSetting.MorphDir  + model + "/";
    std::filesystem::create_directories(morph_dir);
    std::filesystem::create_directories(design_dir);
    spdlog::info("Target dir : {}", target_dir);
    spdlog::info("Param  dir : {}", param_dir);
    spdlog::info("Cond   dir : {}", cond_dir);
    spdlog::info("Design dir : {}", design_dir);
    spdlog::info("Morph  dir : {}", morph_dir);

    // ---- Iter-log CSV root (per-patch files written inside the loop) ------
    // Resources/2_morphlogs/<method>/<model>/patch_<i>_iter_log.csv
    // Schema:
    //   stage,substage,iter,spn,dist,proj_dist,kappa_reg,lambda_reg,penalty_term
    // - substage : WarmOptKap / WarmOptLam / WarmOptP   (warm-up half)
    //              OptKap / OptLam / OptP               (MGDA half)
    // - iter     : SGN inner Newton iter (>=0) or -1 for end-of-substage row
    // - proj_dist: computed at the iter's x state (or final state on -1 rows)
    const std::string morphlogs_dir = config.PathSetting.MorphLogsDir
                                    + config.RuntimeSetting.morph_method + "/"
                                    + model + "/";
    std::filesystem::create_directories(morphlogs_dir);
    spdlog::info("Iter-log dir -> {}", morphlogs_dir);

    // ===== Phase 1: read pre-scaled V/P from disk (ParamAll outputs) =====
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

    // ===== Phase 2: per-patch inverse design =====
    // V/P are already in physical (device) units; no re-scaling here.
    for (auto& pd : patches) {
        const std::string pid = std::to_string(pd.idx);
        spdlog::info("=================================================");
        spdlog::info("Inverse design for patch {} ({} V, {} F)", pd.idx, pd.nV, pd.nF);
        spdlog::info("=================================================");

        Eigen::MatrixXd& V = pd.V;
        Eigen::MatrixXi& F = pd.F;
        Eigen::MatrixXd& P = pd.P;
        const size_t nV = pd.nV;
        const size_t nF = pd.nF;

        ManifoldSurfaceMesh mesh(F);
        VertexPositionGeometry geometry(mesh, V);
        geometry.refreshQuantities();

        std::vector<bool> is_boundary_face;
        std::vector<int> ref_faces = buildRefFaces(mesh, is_boundary_face);

        spdlog::info("Patch {} Step 2: Material settings.", pd.idx);
        Eigen::MatrixXd targetV = V;
        FaceData<Eigen::MatrixXd>   M     = precomputeM(mesh, V, F);
        FaceData<Eigen::Matrix2d>   MrInv = precomputeMrInv(mesh, P, F);

        // Boundary condition: 3 vertex indices from cond file written by ParamAll.
        std::vector<int> fixedVertexIdx;
        {
            const std::string cond_path = cond_dir + "patch_" + pid + "_bound_center.txt";
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
        std::vector<int> fixedIdx;
        for (int v : fixedVertexIdx)
            for (int k = 0; k < 3; ++k) fixedIdx.push_back(3 * v + k);
        std::sort(fixedIdx.begin(), fixedIdx.end());

        const double E  = 1.0;
        const double nu = 0.5;
        Morphmesh morph_mesh(V, P, F, E, nu);
        Morphmesh::ComputeMorphophing(geometry, V, F, nV, nF, ref_faces, MrInv,
                                      morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t,
                                      morph_mesh.kappa_pv_t,  morph_mesh.kappa_pf_t,
                                      &morph_mesh.vertex_area_sum);
        Morphmesh::SetMorphophing(morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t,
                                  morph_mesh.kappa_pv_t,  morph_mesh.kappa_pf_t,
                                  morph_mesh.lambda_pv_s, morph_mesh.lambda_pf_s,
                                  morph_mesh.kappa_pf_s,  morph_mesh.kappa_pf_s);

        igl::writeOBJ(morph_dir + "patch_" + pid + "_targ.obj", V, F);

        VertexData<double> lambda_pv_s(mesh, morph_mesh.lambda_pv_s);
        FaceData<double>   lambda_pf_s(mesh, morph_mesh.lambda_pf_s);
        FaceData<double>   kappa_pf_s (mesh, morph_mesh.kappa_pf_s);

        Eigen::MatrixXd Vr = flatPlateAligned(P, V, fixedVertexIdx);
        igl::writeOBJ(morph_dir + "patch_" + pid + "_init.obj", Vr, F);

        const Eigen::VectorXd masses = computeVertexMasses(geometry);

        // M_kappa depends on MrInv -> refreshed after each P update.
        Eigen::SparseMatrix<double>       M_kappa  = computeFaceMassKappa(mesh, MrInv);
        const Eigen::SparseMatrix<double> M_lambda = computeFaceMassLambda(geometry);
        const Eigen::SparseMatrix<double> L_face   = computeFaceDualLaplacian(mesh);

        // ARAP solver kept as fallback only; active P-update is SGN OptP below.
        LocalGlobalSolver paramSolver(V, F);

        // ---- P-side regularisation matrices for SGN OptP (per patch) -------
        Eigen::SparseMatrix<double> M_P_2(2 * nV, 2 * nV);
        M_P_2.setIdentity();
        Eigen::SparseMatrix<double> L_P_2(2 * nV, 2 * nV);
        {
            Eigen::SparseMatrix<double> L_v;
            igl::cotmatrix(V, F, L_v);
            L_v = (-L_v).eval();
            std::vector<Eigen::Triplet<double>> trips;
            trips.reserve(L_v.nonZeros() * 2);
            for (int i = 0; i < L_v.outerSize(); ++i)
                for (Eigen::SparseMatrix<double>::InnerIterator it(L_v, i); it; ++it)
                    for (int d = 0; d < 2; ++d)
                        trips.emplace_back(2 * (int)it.row() + d, 2 * (int)it.col() + d, it.value());
            L_P_2.setFromTriplets(trips.begin(), trips.end());
        }
        const Eigen::MatrixXd P_anchor = P;

        spdlog::info("Patch {} Step 4: Inverse Design (MGDA).", pd.idx);

        // MGDA does NOT use wP / penalty_threshold; wP_kap / wP_lam ignored.
        const double betaP = config.RuntimeSetting.betaP;

        // === Penalty variant A: 1D independent (current) ===
        auto penalty_to_lamb = MaterialPenaltyFunctionPerF(geometry, ac.feasible_lamb, betaP);
        auto penalty_to_kapp = MaterialPenaltyFunctionPerF(geometry, ac.feasible_kapp, betaP);

        // === Penalty variant B: 2D Euclidean joint (alt; uncomment to switch) ===
        // auto eigen_to_std = [](const Eigen::VectorXd& v) {
        //     return std::vector<double>(v.data(), v.data() + v.size());
        // };
        // auto build_penalty_kap = [&]() {
        //     return MaterialPenaltyFunctionPerF_Joint2D(
        //         geometry, ac.feasible_kapp, ac.feasible_lamb,
        //         eigen_to_std(lambda_pf_s.toVector()),
        //         betaP, /*self_is_kappa=*/true);
        // };
        // auto build_penalty_lam = [&]() {
        //     return MaterialPenaltyFunctionPerF_Joint2D(
        //         geometry, ac.feasible_lamb, ac.feasible_kapp,
        //         eigen_to_std(kappa_pf_s.toVector()),
        //         betaP, /*self_is_kappa=*/false);
        // };

        const int stage_iter = config.RuntimeSetting.stage_iter;
        int k = 0;

        // MGDA: regulariser weights stage-constant.
        // Mutable so the warmup loop can homotopy-decay them between stages.
        double wM_kap = config.RuntimeSetting.wM_kap;
        double wM_lam = config.RuntimeSetting.wM_lam;
        double wL_kap = config.RuntimeSetting.wL_kap;
        double wL_lam = config.RuntimeSetting.wL_lam;

        double distance = 0.0;
        double spn_energy = 0.0;
        double penalty_kap_val = 0.0;
        double penalty_lam_val = 0.0;
        double pareto_kap = 0.0;
        double pareto_lam = 0.0;
        double self_reg = 0.0;

        auto computeKappaReg = [&]() {
            const Eigen::VectorXd kv = kappa_pf_s.toVector();
            return wM_kap * kv.dot(M_kappa * kv) + wL_kap * kv.dot(L_face * kv);
        };
        auto computeLambdaReg = [&]() {
            const Eigen::VectorXd l = lambda_pf_s.toVector();
            return wM_lam * l.dot(M_lambda * l) + wL_lam * l.dot(L_face * l);
        };
        double kappa_reg  = computeKappaReg();
        double lambda_reg = computeLambdaReg();

        // Projected distance: per-face snap -> forward Newton from Vr_start
        // -> mass-weighted dist to target.  Pass a different mesh to score
        // the proj_dist at an SGN inner-iter intermediate state.
        auto computeProjectedDistanceFrom = [&](const Eigen::MatrixXd& Vr_start) -> double {
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
            Eigen::MatrixXd Vr_proj = Vr_start;
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
        auto computeProjectedDistance = [&]() { return computeProjectedDistanceFrom(Vr); };

        // Reshape SGN's flat 3*nV x vector back to nV x 3 V matrix.
        auto reshape_x_to_V = [&](const Eigen::VectorXd& x_vec) {
            Eigen::MatrixXd V_iter(nV, 3);
            for (size_t v = 0; v < nV; ++v)
                for (int j = 0; j < 3; ++j)
                    V_iter(v, j) = x_vec(3 * v + j);
            return V_iter;
        };

        // Re-run forward Newton on the current (P, lambda, kappa) state;
        // updates Vr in place and returns mass-weighted distance.
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

        auto printStageStats = [&]() {
            const double proj_dist     = computeProjectedDistance();
            const double pen_kap_proxy = compute_candidate_diff(ac.feasible_kapp, kappa_pf_s.toVector(), true);
            const double pen_lam_proxy = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
            std::cout << "    patch " << pd.idx
                      << "  SPN energy: " << spn_energy
                      << ", Distance: " << distance
                      << ", Kreg: " << kappa_reg
                      << ", Lreg: " << lambda_reg
                      << ", Projected distance: " << proj_dist
                      << ", Phi_kap: " << penalty_kap_val
                      << ", Phi_lam: " << penalty_lam_val
                      << ", CandDiff_kap: " << pen_kap_proxy
                      << ", CandDiff_lam: " << pen_lam_proxy
                      << ", Pareto_kap: " << pareto_kap
                      << ", Pareto_lam: " << pareto_lam
                      << "\n";
        };

        // ---- Per-patch iter-log CSV (S2_GrayScaleMorph wide-format) -------
        std::ofstream iter_log_ofs(morphlogs_dir + "patch_" + pid + "_iter_log.csv");
        iter_log_ofs << "stage,substage,iter,spn,dist,proj_dist,kappa_reg,lambda_reg,penalty_term\n";
        spdlog::info("Patch {} iter log -> {}", pd.idx,
                     morphlogs_dir + "patch_" + pid + "_iter_log.csv");

        // Each row records the GLOBAL energy decomposition at the current
        // (lambda, kappa) state: Reg_total = kappa_reg + lambda_reg and
        // Phi_total = Phi(kappa) + Phi(lambda), regardless of which variable
        // the SGN substage is currently optimising.
        auto computePhiTotal = [&]() {
            return penalty_to_kapp.eval(kappa_pf_s.toVector())
                 + penalty_to_lamb.eval(lambda_pf_s.toVector());
        };
        auto writeSummaryRow = [&](int stage_idx, const std::string& substage) {
            const double pd_v  = computeProjectedDistance();
            const double kr_v  = computeKappaReg();
            const double lr_v  = computeLambdaReg();
            const double phi_v = computePhiTotal();
            iter_log_ofs << stage_idx << "," << substage << ",-1,"
                         << spn_energy << "," << distance << "," << pd_v << ","
                         << kr_v << "," << lr_v << "," << phi_v << "\n";
            iter_log_ofs.flush();
        };
        auto makeIterLogger = [&](int stage_idx, const std::string& substage) -> SgnIterCallback {
            return [&, stage_idx, substage]
                   (int iter, const Eigen::VectorXd& x_iter,
                    double spn, double dist_v, double /*self_reg*/, double /*penalty*/) {
                const double pd_v  = computeProjectedDistanceFrom(reshape_x_to_V(x_iter));
                const double kr_v  = computeKappaReg();
                const double lr_v  = computeLambdaReg();
                const double phi_v = computePhiTotal();
                iter_log_ofs << stage_idx << "," << substage << "," << iter << ","
                             << spn << "," << dist_v << "," << pd_v << ","
                             << kr_v << "," << lr_v << "," << phi_v << "\n";
            };
        };

        // SGN OptP with snap-material flow (port from develop_homotopy 41b4187).
        // 1) snap (lambda, kappa) to nearest feasible candidate
        // 2) SGN OptP on snapped material -> SGN distance == proj_dist
        // 3) hard-snap continuous lambda_pf_s/kappa_pf_s to same candidates
        auto runArapPUpdate = [&](const std::string& tag) {
            if (!config.RuntimeSetting.enable_optp) {
                distance   = recomputeForwardState();
                kappa_reg  = computeKappaReg();
                lambda_reg = computeLambdaReg();
                spn_energy = distance + kappa_reg + lambda_reg;
                std::cout << "[OptP skipped] patch " << pd.idx << " " << tag << "  ";
                printStageStats();
                return;
            }
            const double wM_P = config.RuntimeSetting.wM_P;
            const double wL_P = config.RuntimeSetting.wL_P;

            FaceData<double> lambda_pf_snap(mesh);
            FaceData<double> kappa_pf_snap(mesh);
            for (Face f : mesh.faces()) {
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb,
                                            kappa_pf_s[f], lambda_pf_s[f]);
                lambda_pf_snap[f] = ac.feasible_lamb[idx];
                kappa_pf_snap[f]  = ac.feasible_kapp[idx];
            }

            auto adjointFunc_OptP = adjointFunction_FixMaterial_OptP(
                geometry, F, lambda_pf_snap, kappa_pf_snap,
                E, nu, ac.thickness,
                config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            double P_reg = 0.0;
            Vr = sparse_gauss_newton_FixMaterial_OptP(
                geometry, F, targetV, Vr, P,
                lambda_pf_snap, kappa_pf_snap,
                masses, M_P_2, L_P_2, P_anchor,
                kappa_reg + lambda_reg,
                adjointFunc_OptP, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                wM_P, wL_P,
                E, nu, ac.thickness,
                config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                distance, spn_energy, P_reg);

            for (Face f : mesh.faces()) {
                lambda_pf_s[f] = lambda_pf_snap[f];
                kappa_pf_s[f]  = kappa_pf_snap[f];
            }

            Eigen::MatrixXd P_obj(P.rows(), 3);
            P_obj.leftCols(2) = P;
            P_obj.col(2).setZero();
            igl::writeOBJ(morph_dir + "patch_" + pid + "_P_" + tag + ".obj", P_obj, F);
            MrInv   = precomputeMrInv(mesh, P, F);
            M_kappa = computeFaceMassKappa(mesh, MrInv);

            distance   = recomputeForwardState();
            kappa_reg  = computeKappaReg();
            lambda_reg = computeLambdaReg();
            spn_energy = distance + kappa_reg + lambda_reg;

            const Eigen::VectorXd lambdaVec_now = lambda_pf_s.toVector();
            std::cout << "[OptP finish] patch " << pd.idx << " " << tag
                      << ": lambda range [" << lambdaVec_now.minCoeff()
                      << ", " << lambdaVec_now.maxCoeff() << "]  ";
            printStageStats();
        };

        // ---- Warm-up: pure-SPN SGN (no penalty) ----
        const int warmup_stages = config.RuntimeSetting.warmup_stages;
        for (int kw = 0; kw < warmup_stages; ++kw) {
            printf("============================ patch %zu Warmup stage %d (pure SPN) ============================\n",
                   pd.idx, kw);
            std::cout << "Parameters Settings (Regular):  wM_kap = " << wM_kap << ", wL_kap = " << wL_kap
                      << ", wM_lam = " << wM_lam << ", wL_lam = " << wL_lam << "\n";

            printf("---- patch %zu Warmup OptKap (no penalty) ----\n", pd.idx);
            auto adjF_w_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness,
                                                                config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixLam_OptKap(
                     geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, lambda_reg,
                     adjF_w_OptKap, fixedIdx,
                     config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                     wM_kap, wL_kap,
                     E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                     distance, spn_energy, self_reg,
                     [](const auto&){},
                     makeIterLogger(kw, "WarmOptKap"));
            kappa_reg = self_reg;
            printStageStats();
            writeSummaryRow(kw, "WarmOptKap");

            printf("---- patch %zu Warmup OptLam (no penalty) ----\n", pd.idx);
            auto adjF_w_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness,
                                                                 config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixKap_OptLam(
                     geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, kappa_reg,
                     adjF_w_OptLam, fixedIdx,
                     config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                     wM_lam, wL_lam,
                     E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                     distance, spn_energy, self_reg,
                     [](const auto&){},
                     makeIterLogger(kw, "WarmOptLam"));
            lambda_reg = self_reg;
            printStageStats();
            writeSummaryRow(kw, "WarmOptLam");

            runArapPUpdate("warmup" + std::to_string(kw));
            // runArapPUpdate already refreshed kappa_reg / lambda_reg / spn_energy.
            writeSummaryRow(kw, "WarmOptP");

            // Homotopy-style decay of reg coefficients between warmup stages.
            const double decay = config.RuntimeSetting.warmup_reg_decay;
            wM_kap *= decay;  wL_kap *= decay;
            wM_lam *= decay;  wL_lam *= decay;
            kappa_reg  = computeKappaReg();
            lambda_reg = computeLambdaReg();
        }
        printf("============================ patch %zu Warmup done, entering MGDA ============================\n",
               pd.idx);

        // ---- Trust-region safeguard state -------------------------------------
        double dist_best = std::numeric_limits<double>::infinity();
        double proj_best = std::numeric_limits<double>::infinity();
        Eigen::MatrixXd                Vr_best        = Vr;
        FaceData<double>               lambda_pf_best = lambda_pf_s;
        FaceData<double>               kappa_pf_best  = kappa_pf_s;
        Eigen::MatrixXd                P_best         = P;
        FaceData<Eigen::Matrix2d>      MrInv_best     = MrInv;
        Eigen::SparseMatrix<double>    M_kappa_best   = M_kappa;
        double kappa_reg_best = kappa_reg, lambda_reg_best = lambda_reg;

        while (k < stage_iter) {
            printf("------------------------- patch %zu Stage: %d (MGDA) -------------------------\n", pd.idx, k);
            std::cout << "Parameters Settings (Regular):  wM_kap = " << wM_kap << ", wL_kap = " << wL_kap
                      << ", wM_lam = " << wM_lam << ", wL_lam = " << wL_lam
                      << ", betaP = " << betaP << "\n";

            printf("---------------------- patch %zu OptKap Start (MGDA) ----------------------\n", pd.idx);
            auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness,
                                                                      config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            // 1D penalty path: cand_other / other_const default-empty -> SGN d_P = 1D snap.
            // Power-law alpha schedule.  exp=1 linear; exp>1 stays near start longer.
            const double a_start = config.RuntimeSetting.mgda_alpha_start;
            const double a_end   = config.RuntimeSetting.mgda_alpha_end;
            const double a_exp   = config.RuntimeSetting.mgda_alpha_decay_exp;
            double alpha_k = -1.0;
            if (a_start >= 0.0 && a_end >= 0.0 && stage_iter > 1) {
                const double t = double(k) / double(stage_iter - 1);
                alpha_k = a_end + (a_start - a_end) * std::pow(1.0 - t, a_exp);
            }
            Vr = sparse_gauss_newton_FixLam_OptKap_MGDA(
                     geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, lambda_reg,
                     adjointFunc_OptKap, penalty_to_kapp,
                     ac.feasible_kapp, betaP, fixedIdx,
                     config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                     wM_kap, wL_kap,
                     E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                     distance, spn_energy, self_reg, penalty_kap_val, pareto_kap,
                     [](const auto&){},
                     makeIterLogger(k, "OptKap"),
                     /*cand_other=*/{}, /*other_const=*/{},
                     /*alpha_override=*/alpha_k);
            kappa_reg = self_reg;
            printStageStats();
            writeSummaryRow(k, "OptKap");
            printf("---------------------- patch %zu OptKap Finish ----------------------\n", pd.idx);

            printf("---------------------- patch %zu OptLam Start (MGDA) ----------------------\n", pd.idx);
            auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness,
                                                                       config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixKap_OptLam_MGDA(
                     geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, kappa_reg,
                     adjointFunc_OptLam, penalty_to_lamb,
                     ac.feasible_lamb, betaP, fixedIdx,
                     config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                     wM_lam, wL_lam,
                     E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                     distance, spn_energy, self_reg, penalty_lam_val, pareto_lam,
                     [](const auto&){},
                     makeIterLogger(k, "OptLam"),
                     /*cand_other=*/{}, /*other_const=*/{},
                     /*alpha_override=*/alpha_k);
            lambda_reg = self_reg;
            printStageStats();
            writeSummaryRow(k, "OptLam");
            printf("---------------------- patch %zu OptLam Finish ----------------------\n", pd.idx);

            runArapPUpdate("stage" + std::to_string(k));
            // runArapPUpdate already refreshed kappa_reg / lambda_reg / spn_energy.
            writeSummaryRow(k, "OptP");

            // ---- Trust-region: best snapshot tracks historical min proj_dist;
            //      reject (revert) only when BOTH dist and proj worsen.  See
            //      develop_homotopy 0d42dfd. ----
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
                kappa_reg   = kappa_reg_best;
                lambda_reg  = lambda_reg_best;
                distance    = dist_best;
                std::cout << "[REJECT] patch " << pd.idx << " stage " << k
                          << ": dist=" << dist_new << ">" << dist_best
                          << " AND proj=" << proj_new << ">" << proj_best
                          << "; revert\n";
            } else if (snapshot_improves) {
                std::cout << "[ACCEPT, best updated] patch " << pd.idx << " stage " << k
                          << ": dist=" << dist_new << "  proj=" << proj_new << " (best now)\n";
            } else {
                std::cout << "[ACCEPT, best unchanged] patch " << pd.idx << " stage " << k
                          << ": dist=" << dist_new << "  proj=" << proj_new
                          << " (best proj still " << proj_best << ")\n";
            }
            // -------------------------------------------------------------------

            k++;

            // Homotopy-style reg decay between MGDA stages: F = dist + reg
            // slowly slims toward F = dist.  Default 1.0 = no decay.
            {
                const double mgda_decay = config.RuntimeSetting.mgda_reg_decay;
                wM_kap *= mgda_decay;  wL_kap *= mgda_decay;
                wM_lam *= mgda_decay;  wL_lam *= mgda_decay;
                kappa_reg  = computeKappaReg();
                lambda_reg = computeLambdaReg();
            }

            printf("--------------------------------------------------------------------------\n");
        }

        // Restore best snapshot for downstream material output + proj.obj.
        Vr          = Vr_best;
        lambda_pf_s = lambda_pf_best;
        kappa_pf_s  = kappa_pf_best;
        P           = P_best;
        MrInv       = MrInv_best;
        M_kappa     = M_kappa_best;
        std::cout << "Patch " << pd.idx << " restored best snapshot: dist=" << dist_best
                  << "  proj=" << proj_best << "\n";

        // ---- Optional one-shot final OptP on snapped material (per patch) ----
        if (config.RuntimeSetting.enable_final_optp) {
            printf("======== patch %zu Final OptP (snap-material) ========\n", pd.idx);
            const double wM_P = config.RuntimeSetting.wM_P;
            const double wL_P = config.RuntimeSetting.wL_P;

            FaceData<double> lambda_pf_snap(mesh);
            FaceData<double> kappa_pf_snap(mesh);
            for (Face f : mesh.faces()) {
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb,
                                            kappa_pf_s[f], lambda_pf_s[f]);
                lambda_pf_snap[f] = ac.feasible_lamb[idx];
                kappa_pf_snap[f]  = ac.feasible_kapp[idx];
            }
            auto adjointFunc_OptP = adjointFunction_FixMaterial_OptP(
                geometry, F, lambda_pf_snap, kappa_pf_snap,
                E, nu, ac.thickness,
                config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            double P_reg = 0.0;
            Vr = sparse_gauss_newton_FixMaterial_OptP(
                geometry, F, targetV, Vr, P,
                lambda_pf_snap, kappa_pf_snap,
                masses, M_P_2, L_P_2, P_anchor,
                kappa_reg + lambda_reg,
                adjointFunc_OptP, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                wM_P, wL_P,
                E, nu, ac.thickness,
                config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                distance, spn_energy, P_reg,
                makeIterLogger(stage_iter, "FinalOptP"));

            for (Face f : mesh.faces()) {
                lambda_pf_s[f] = lambda_pf_snap[f];
                kappa_pf_s[f]  = kappa_pf_snap[f];
            }
            MrInv   = precomputeMrInv(mesh, P, F);
            M_kappa = computeFaceMassKappa(mesh, MrInv);
            distance   = recomputeForwardState();
            kappa_reg  = computeKappaReg();
            lambda_reg = computeLambdaReg();
            spn_energy = distance + kappa_reg + lambda_reg;
            writeSummaryRow(stage_iter, "FinalOptP");
            std::cout << "[Final OptP finish] patch " << pd.idx
                      << " dist=" << distance
                      << "  proj=" << computeProjectedDistance() << "\n";

            Eigen::MatrixXd P_obj(P.rows(), 3);
            P_obj.leftCols(2) = P;
            P_obj.col(2).setZero();
            igl::writeOBJ(morph_dir + "patch_" + pid + "_P_final.obj", P_obj, F);
        }

        igl::writeOBJ(morph_dir + "patch_" + pid + "_inv.obj", Vr, F);

        // ---- Final projected forward sim (manufacturing reality) ----------
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
            const std::string proj_path = morph_dir + "patch_" + pid + "_proj.obj";
            igl::writeOBJ(proj_path, Vr_proj, F);
            spdlog::info("Patch {} proj mesh -> {}", pd.idx, proj_path);
        }

        // ---- Material projection: per-face (lambda, kappa) -> nearest feasible (t1, t2) ----
        {
            const std::string mat_path = design_dir + "patch_" + pid + "_material.txt";
            const std::string lk_path  = design_dir + "patch_" + pid + "_lamkap.txt";
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

        std::cout << "==========================================================\n";
        std::cout << "  Patch " << pd.idx
                  << " FINAL Projected distance (manufactured design):  "
                  << final_proj_dist << "\n";
        std::cout << "==========================================================\n";

        spdlog::info("Patch {} done.", pd.idx);
    }

    spdlog::info("InverseAll: all patches processed.");
    return 0;
}
