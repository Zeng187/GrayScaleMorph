

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
        const Eigen::SparseMatrix<double> M_kappa = computeFaceMassKappa(mesh, MrInv);
        const Eigen::SparseMatrix<double> M_lambda = computeFaceMassLambda(geometry);
        const Eigen::SparseMatrix<double> L_face = computeFaceDualLaplacian(mesh);

        spdlog::info("Step 4: Inverse Design.");

        double wP_kap = config.RuntimeSetting.wP_kap;
        double wP_lam = config.RuntimeSetting.wP_lam;
        double penalty_threshold = config.RuntimeSetting.penalty_threshold;
        double betaP = config.RuntimeSetting.betaP;
        auto penalty_to_lamb = MaterialPenaltyFunctionPerF(geometry, ac.feasible_lamb, betaP);
        auto penalty_to_kapp = MaterialPenaltyFunctionPerF(geometry, ac.feasible_kapp, betaP);
        auto penalty_to_modu = MaterialPenaltyFunctionPerV(geometry, ac.feasible_modl, betaP);

        int stage_iter = 5;
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
        while(k < stage_iter)
        {
            spdlog::info("Patch {} Stage {}, OptKap start, wP_kap: {:.6f}, wP_lam: {:.6f}.", pd.idx, k, wP_kap, wP_lam);
            auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixLam_OptKap_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, lambda_reg,
                adjointFunc_OptKap, penalty_to_kapp, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_kap, wL_kap, wP_kap,
                E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces,
                distance, spn_energy, self_reg);
            kappa_reg = self_reg;

            penalty_kap = compute_candidate_diff(ac.feasible_kapp,kappa_pf_s.toVector(),true);
            penalty_lam = compute_candidate_diff(ac.feasible_lamb,lambda_pf_s.toVector(),true);
            spdlog::info("Patch {} Stage {}, OptKap finish - Distance: {:.6f}, SPN energy: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                         pd.idx, k, distance, spn_energy, penalty_kap, penalty_lam);



            spdlog::info("Patch {} Stage {}, OptLam start, wP_kap: {:.6f}, wP_lam: {:.6f}.", pd.idx, k, wP_kap, wP_lam);
            auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixKap_OptLam_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, kappa_reg,
                adjointFunc_OptLam, penalty_to_lamb, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_lam, wL_lam, wP_lam,
                E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces,
                distance, spn_energy, self_reg);
            lambda_reg = self_reg;

            penalty_kap = compute_candidate_diff(ac.feasible_kapp,kappa_pf_s.toVector(),true);
            penalty_lam = compute_candidate_diff(ac.feasible_lamb,lambda_pf_s.toVector(),true);
            spdlog::info("Patch {} Stage {}, OptLam finish- Distance: {:.6f}, SPN energy: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                         pd.idx, k, distance, spn_energy, penalty_kap, penalty_lam);

            // Evaluate distance after jointly projecting kappa and lambda to the same feasible index
            {
                FaceData<double> kappa_pf_proj(mesh);
                FaceData<double> lambda_pf_proj(mesh);

                for (Face f : mesh.faces()) {
                    double kap = kappa_pf_s[f];
                    double lam = lambda_pf_s[f];
                    int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap, lam);
                    kappa_pf_proj[f] = ac.feasible_kapp[idx];
                    lambda_pf_proj[f] = ac.feasible_lamb[idx];
                }

                auto simFunc_proj = simulationFunction(geometry, MrInv, lambda_pf_proj, kappa_pf_proj,
                    E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
                Eigen::MatrixXd Vr_proj = Vr;
                newton(geometry, Vr_proj, simFunc_proj,
                    config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx);
                double dist_proj = (Vr_proj - targetV).squaredNorm() / nV;

                spdlog::info("Patch {} Stage {}, Projected distance: {:.6f}", pd.idx, k, dist_proj);
            }

            k++;
            if (penalty_kap >= penalty_threshold) {
                wP_kap *= 10;
            }
            if (penalty_lam >= penalty_threshold) {
                wP_lam *= 10;
            }

            if(penalty_kap < penalty_threshold && penalty_lam < penalty_threshold)
                break;
        }


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
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, config.RuntimeSetting.wM, config.RuntimeSetting.wL,
                E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces,
                distance_kap, spn_kap, self_reg_kap);
            kappa_reg_np = self_reg_kap;

            spdlog::info("Patch {} Stage {}, OptKap finish - Distance: {:.6f}, SPN energy: {:.6f}", pd.idx, k, distance_kap, spn_kap);


            spdlog::info("Patch {} Stage {}, OptLam start", pd.idx, k);
            auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixKap_OptLam(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, kappa_reg_np,
                adjointFunc_OptLam, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, 0.0, config.RuntimeSetting.wL,
                E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces,
                distance_lam, spn_lam, self_reg_lam);
            lambda_reg_np = self_reg_lam;

            spdlog::info("Patch {} Stage {}, OptLam finish - Distance: {:.6f}, SPN energy: {:.6f}", pd.idx, k, distance_lam, spn_lam);


            k++;
        }


#endif



        // Vr lives in the same physical frame as V; write as-is.
        igl::writeOBJ(morph_dir + "patch_" + std::to_string(pd.idx) + "_inv.obj", Vr, F);

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

        spdlog::info("Patch {} done.", pd.idx);
    }

    spdlog::info("All patches processed; program finish.");

}

