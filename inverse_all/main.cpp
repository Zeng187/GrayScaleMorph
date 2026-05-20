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
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
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

        // ARAP solver for lambda-aware P-update at each stage end.
        LocalGlobalSolver paramSolver(V, F);

        spdlog::info("Patch {} Step 4: Inverse Design (MGDA).", pd.idx);

        // MGDA does NOT use wP / penalty_threshold; wP_kap / wP_lam ignored.
        const double betaP = config.RuntimeSetting.betaP;
        auto penalty_to_lamb = MaterialPenaltyFunctionPerF(geometry, ac.feasible_lamb, betaP);
        auto penalty_to_kapp = MaterialPenaltyFunctionPerF(geometry, ac.feasible_kapp, betaP);

        const int stage_iter = 5;
        int k = 0;

        // MGDA: regulariser weights stage-constant.
        const double wM_kap = config.RuntimeSetting.wM_kap;
        const double wM_lam = config.RuntimeSetting.wM_lam;
        const double wL_kap = config.RuntimeSetting.wL_kap;
        const double wL_lam = config.RuntimeSetting.wL_lam;

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

        // Projected distance: per-face snap (lambda, kappa) -> nearest feasible
        // material, re-run forward Newton, mass-weighted dist to target.
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

        auto printStageStats = [&]() {
            const double proj_dist     = computeProjectedDistance();
            const double pen_kap_proxy = compute_candidate_diff(ac.feasible_kapp, kappa_pf_s.toVector(), true);
            const double pen_lam_proxy = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
            std::cout << "    patch " << pd.idx
                      << "  SPN energy: " << spn_energy
                      << ", Distance: " << distance
                      << ", Projected distance: " << proj_dist
                      << ", Phi_kap: " << penalty_kap_val
                      << ", Phi_lam: " << penalty_lam_val
                      << ", CandDiff_kap: " << pen_kap_proxy
                      << ", CandDiff_lam: " << pen_lam_proxy
                      << ", Pareto_kap: " << pareto_kap
                      << ", Pareto_lam: " << pareto_lam
                      << "\n";
        };

        // ARAP P-update (reused by warm-up and MGDA stages).
        auto runArapPUpdate = [&](const std::string& tag) {
            Eigen::VectorXd lambdaVec = lambda_pf_s.toVector();
            Eigen::VectorXd sTarget = 1.0 / lambdaVec.array();
            Eigen::MatrixX2d P_2d = P;
            paramSolver.solve(P_2d, sTarget, sTarget, 10);
            P = P_2d;
            Eigen::MatrixXd P_obj(P.rows(), 3);
            P_obj.leftCols(2) = P;
            P_obj.col(2).setZero();
            igl::writeOBJ(morph_dir + "patch_" + pid + "_P_" + tag + ".obj", P_obj, F);
            MrInv   = precomputeMrInv(mesh, P, F);
            M_kappa = computeFaceMassKappa(mesh, MrInv);
            std::cout << "[P-update] patch " << pd.idx << " " << tag
                      << ": lambda range [" << lambdaVec.minCoeff()
                      << ", " << lambdaVec.maxCoeff() << "]\n";
        };

        // ---- Warm-up: pure-SPN SGN (no penalty) ----
        const int warmup_stages = 1;
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
                     distance, spn_energy, self_reg);
            kappa_reg = self_reg;
            printStageStats();

            printf("---- patch %zu Warmup OptLam (no penalty) ----\n", pd.idx);
            auto adjF_w_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness,
                                                                 config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixKap_OptLam(
                     geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, kappa_reg,
                     adjF_w_OptLam, fixedIdx,
                     config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                     wM_lam, wL_lam,
                     E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                     distance, spn_energy, self_reg);
            lambda_reg = self_reg;
            printStageStats();

            runArapPUpdate("warmup" + std::to_string(kw));
            kappa_reg  = computeKappaReg();
            lambda_reg = computeLambdaReg();
        }
        printf("============================ patch %zu Warmup done, entering MGDA ============================\n",
               pd.idx);

        while (k < stage_iter) {
            printf("------------------------- patch %zu Stage: %d (MGDA) -------------------------\n", pd.idx, k);
            std::cout << "Parameters Settings (Regular):  wM_kap = " << wM_kap << ", wL_kap = " << wL_kap
                      << ", wM_lam = " << wM_lam << ", wL_lam = " << wL_lam
                      << ", betaP = " << betaP << "\n";

            printf("---------------------- patch %zu OptKap Start (MGDA) ----------------------\n", pd.idx);
            auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness,
                                                                      config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixLam_OptKap_MGDA(
                     geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, lambda_reg,
                     adjointFunc_OptKap, penalty_to_kapp, ac.feasible_kapp, betaP, fixedIdx,
                     config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                     wM_kap, wL_kap,
                     E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                     distance, spn_energy, self_reg, penalty_kap_val, pareto_kap);
            kappa_reg = self_reg;
            printStageStats();
            printf("---------------------- patch %zu OptKap Finish ----------------------\n", pd.idx);

            printf("---------------------- patch %zu OptLam Start (MGDA) ----------------------\n", pd.idx);
            auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness,
                                                                       config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixKap_OptLam_MGDA(
                     geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, kappa_reg,
                     adjointFunc_OptLam, penalty_to_lamb, ac.feasible_lamb, betaP, fixedIdx,
                     config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                     wM_lam, wL_lam,
                     E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                     distance, spn_energy, self_reg, penalty_lam_val, pareto_lam);
            lambda_reg = self_reg;
            printStageStats();
            printf("---------------------- patch %zu OptLam Finish ----------------------\n", pd.idx);

            runArapPUpdate("stage" + std::to_string(k));

            k++;

            // MGDA termination: both stages reached Pareto criticality.
            if (pareto_kap < config.RuntimeSetting.epsilon &&
                pareto_lam < config.RuntimeSetting.epsilon)
            {
                std::cout << "[MGDA] patch " << pd.idx
                          << " both stages Pareto-critical, stopping at k=" << k << "\n";
                break;
            }

            kappa_reg  = computeKappaReg();
            lambda_reg = computeLambdaReg();

            printf("--------------------------------------------------------------------------\n");
        }

        igl::writeOBJ(morph_dir + "patch_" + pid + "_inv.obj", Vr, F);

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

        spdlog::info("Patch {} done.", pd.idx);
    }

    spdlog::info("InverseAll: all patches processed.");
    return 0;
}
