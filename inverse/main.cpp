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

    // ARAP solver for the lambda-aware P-update done at each stage end.
    // Initialised once from (V, F): cotmatrix / factorisation is reused
    // across all stages since V topology doesn't change.
    LocalGlobalSolver paramSolver(V, F);

    spdlog::info("Step 4: Inverse Design (MGDA).");

    // MGDA does NOT use wP / penalty_threshold (penalty is the second objective,
    // not a weighted term).  wP_kap / wP_lam in cfg.json are read but ignored.
    double betaP = config.RuntimeSetting.betaP;
    auto penalty_to_lamb = MaterialPenaltyFunctionPerF(geometry, ac.feasible_lamb, betaP);
    auto penalty_to_kapp = MaterialPenaltyFunctionPerF(geometry, ac.feasible_kapp, betaP);

    int stage_iter = 5;
    int k = 0;

    // MGDA also drops wM/wL stage decay: regulariser weights are stage-constant
    // and only meaningful inside F (their gradient enters d_F).
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

    auto computeKappaReg = [&]()
    {
        const Eigen::VectorXd kv = kappa_pf_s.toVector();
        return wM_kap * kv.dot(M_kappa * kv) + wL_kap * kv.dot(L_face * kv);
    };
    auto computeLambdaReg = [&]()
    {
        const Eigen::VectorXd l = lambda_pf_s.toVector();
        return wM_lam * l.dot(M_lambda * l) + wL_lam * l.dot(L_face * l);
    };
    double kappa_reg = computeKappaReg();
    double lambda_reg = computeLambdaReg();

    // Projected distance: snap the current (kappa, lambda) on every face to the
    // nearest feasible material pair, re-run forward Newton, and return the
    // mass-weighted distance from the resulting Vr_proj to the target.  This
    // is what the discretised material design will actually produce.
    auto computeProjectedDistance = [&]() -> double
    {
        FaceData<double> kappa_pf_proj(mesh);
        FaceData<double> lambda_pf_proj(mesh);
        for (Face f : mesh.faces())
        {
            int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb,
                                        kappa_pf_s[f], lambda_pf_s[f]);
            kappa_pf_proj[f] = ac.feasible_kapp[idx];
            lambda_pf_proj[f] = ac.feasible_lamb[idx];
        }
        auto simFunc_proj = simulationFunction(geometry, MrInv, lambda_pf_proj, kappa_pf_proj,
                                               E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Eigen::MatrixXd Vr_proj = Vr;
        newton(geometry, Vr_proj, simFunc_proj,
               config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx);
        double d2 = 0.0;
        for (size_t i = 0; i < nV; ++i)
            for (int j = 0; j < 3; ++j)
            {
                double d = Vr_proj(i, j) - targetV(i, j);
                d2 += masses(3 * i + j) * d * d;
            }
        return d2;
    };

    // MGDA stats: SPN energy / distance / projected dist / penalties / Pareto norms.
    auto printStageStats = [&]()
    {
        const double proj_dist = computeProjectedDistance();
        const double pen_kap_proxy = compute_candidate_diff(ac.feasible_kapp, kappa_pf_s.toVector(), true);
        const double pen_lam_proxy = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
        std::cout << "SPN energy: " << spn_energy
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

    // ARAP P-update lambda (reused by warm-up and MGDA stages).
    auto runArapPUpdate = [&](const std::string& tag) {
        Eigen::VectorXd lambdaVec = lambda_pf_s.toVector();
        Eigen::VectorXd sTarget = 1.0 / lambdaVec.array();
        Eigen::MatrixX2d P_2d = P;
        paramSolver.solve(P_2d, sTarget, sTarget, 10);
        P = P_2d;
        Eigen::MatrixXd P_obj(P.rows(), 3);
        P_obj.leftCols(2) = P;
        P_obj.col(2).setZero();
        igl::writeOBJ(morph_dir + "patch_0_P_" + tag + ".obj", P_obj, F);
        MrInv = precomputeMrInv(mesh, P, F);
        M_kappa = computeFaceMassKappa(mesh, MrInv);
        std::cout << "[P-update] " << tag
                  << ": lambda range [" << lambdaVec.minCoeff()
                  << ", " << lambdaVec.maxCoeff() << "]\n";
    };

    // ---- Warm-up: pure-SPN SGN (no penalty) so distance/projected-distance
    //      drop into a sensible basin before MGDA wakes Phi as 2nd objective.
    const int warmup_stages = 1;
    for (int kw = 0; kw < warmup_stages; ++kw) {
        printf("============================ Warmup stage %d (pure SPN) ============================\n", kw);
        std::cout << "Parameters Settings (Regular):  wM_kap = " << wM_kap << ", wL_kap = " << wL_kap
                  << ", wM_lam = " << wM_lam << ", wL_lam = " << wL_lam << "\n";

        printf("---- Warmup OptKap (no penalty) ----\n");
        auto adjF_w_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Vr = sparse_gauss_newton_FixLam_OptKap(
                 geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, lambda_reg,
                 adjF_w_OptKap, fixedIdx,
                 config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                 wM_kap, wL_kap,
                 E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                 distance, spn_energy, self_reg);
        kappa_reg = self_reg;
        printStageStats();

        printf("---- Warmup OptLam (no penalty) ----\n");
        auto adjF_w_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
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
        // Reg accumulators after M_kappa refresh.
        kappa_reg = computeKappaReg();
        lambda_reg = computeLambdaReg();
    }
    printf("============================ Warmup done, entering MGDA ============================\n");

    while (k < stage_iter)
    {
        printf("------------------------------------------------------ Stage: %d (MGDA) ------------------------------------------------------\n", k);
        std::cout << "Parameters Settings (Regular):  wM_kap = " << wM_kap << ", wL_kap = " << wL_kap
                  << ", wM_lam = " << wM_lam << ", wL_lam = " << wL_lam
                  << ", betaP = " << betaP << "\n";

        printf("----------------------------  OptKap Start (MGDA) ----------------------------\n", k);
        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Vr = sparse_gauss_newton_FixLam_OptKap_MGDA(
                 geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, lambda_reg,
                 adjointFunc_OptKap, penalty_to_kapp, ac.feasible_kapp, betaP, fixedIdx,
                 config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                 wM_kap, wL_kap,
                 E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                 distance, spn_energy, self_reg, penalty_kap_val, pareto_kap);
        kappa_reg = self_reg;
        printStageStats();
        printf("----------------------------  OptKap Finish ----------------------------\n", k);

        printf("----------------------------  OptLam Start (MGDA) ----------------------------\n", k);
        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Vr = sparse_gauss_newton_FixKap_OptLam_MGDA(
                 geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, kappa_reg,
                 adjointFunc_OptLam, penalty_to_lamb, ac.feasible_lamb, betaP, fixedIdx,
                 config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                 wM_lam, wL_lam,
                 E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                 distance, spn_energy, self_reg, penalty_lam_val, pareto_lam);
        lambda_reg = self_reg;
        printStageStats();
        printf("----------------------------  OptLam Finish ----------------------------\n", k);

        runArapPUpdate("stage" + std::to_string(k));

        k++;

        // MGDA termination: both stages reached Pareto criticality.
        if (pareto_kap < config.RuntimeSetting.epsilon &&
            pareto_lam < config.RuntimeSetting.epsilon)
        {
            std::cout << "[MGDA] both stages Pareto-critical, stopping at k=" << k << "\n";
            break;
        }

        // Refresh kappa_reg after M_kappa update (MrInv changed).
        kappa_reg = computeKappaReg();
        lambda_reg = computeLambdaReg();

        printf("----------------------------------------------------------------------------------------------------------------------\n");
    }

    // V_target was already in physical (device) units; Vr lives in the same
    // frame, so write it out as-is — no inverse rescaling.
    igl::writeOBJ(morph_dir + "patch_0_inv.obj", Vr, F);

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

    spdlog::info("Inverse (single mesh): done.  Output -> {}", morph_dir);
    return 0;
}
