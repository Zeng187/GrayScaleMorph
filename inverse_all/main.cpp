

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
            morph_mesh.kappa_pv_s, morph_mesh.kappa_pf_s);

        // Mirror physical-unit target into morph_dir so {targ, inv} sit side-by-side.
        igl::writeOBJ(morph_dir + "patch_" + std::to_string(pd.idx) + "_targ.obj", V, F);


        VertexData<double> lambda_pv_s(mesh, morph_mesh.lambda_pv_s);
        VertexData<double> kappa_pv_s(mesh, morph_mesh.kappa_pv_s);
        FaceData<double> lambda_pf_s(mesh, morph_mesh.lambda_pf_s);
        FaceData<double> kappa_pf_s(mesh, morph_mesh.kappa_pf_s);

        ///***************************************** Inverse Design *****************************************///

        auto V_pred = V, Vr = V;

        spdlog::info("Patch {} Inverse Design (ADMM + Augmented Lagrangian).", pd.idx);

        const double wM_kap = config.RuntimeSetting.wM_kap;
        const double wM_lam = config.RuntimeSetting.wM_lam;
        const double wL_kap = config.RuntimeSetting.wL_kap;
        const double wL_lam = config.RuntimeSetting.wL_lam;

        // ADMM hyperparameters (shared by all patches)
        double       rho            = config.RuntimeSetting.rho;
        const double rho_max        = config.RuntimeSetting.rho_max;
        const double rho_growth     = config.RuntimeSetting.rho_growth;
        const double rho_ratio      = config.RuntimeSetting.rho_ratio;
        const int    max_outer_iter = config.RuntimeSetting.max_outer_iter;
        const double tol_primal     = config.RuntimeSetting.tol_primal;
        const double tol_dual       = config.RuntimeSetting.tol_dual;

        FaceData<double> z_lam(mesh, 0.0), z_kap(mesh, 0.0);
        FaceData<double> mu_lam(mesh, 0.0), mu_kap(mesh, 0.0);

        auto avg_kap_pf = [&](Face f) {
            double sum = 0.0; int cnt = 0;
            for (Vertex v : f.adjacentVertices()) { sum += kappa_pv_s[v]; cnt++; }
            return sum / cnt;
        };

        for (Face f : mesh.faces()) {
            double kap_f = avg_kap_pf(f);
            double lam_f = lambda_pf_s[f];
            int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap_f, lam_f);
            z_lam[f] = ac.feasible_lamb[idx];
            z_kap[f] = ac.feasible_kapp[idx];
        }

        for (int outer = 0; outer < max_outer_iter; ++outer)
        {
            spdlog::info("Patch {} ADMM iter {} start (rho={:.4f})", pd.idx, outer, rho);

            auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixLam_OptKap_AL(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s,
                adjointFunc_OptKap, z_kap, mu_kap, rho, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                wM_kap, wL_kap, E, nu, ac.thickness,
                config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            double dist_after_kap = (Vr - targetV).squaredNorm() / nV;

            auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pv_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixKap_OptLam_AL(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s,
                adjointFunc_OptLam, z_lam, mu_lam, rho, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                wM_lam, wL_lam, E, nu, ac.thickness,
                config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            double dist_after_lam = (Vr - targetV).squaredNorm() / nV;

            FaceData<double> z_lam_old = z_lam, z_kap_old = z_kap;
            for (Face f : mesh.faces()) {
                double kap_f = avg_kap_pf(f);
                double lam_f = lambda_pf_s[f];
                double kap_shifted = kap_f + mu_kap[f] / rho;
                double lam_shifted = lam_f + mu_lam[f] / rho;
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap_shifted, lam_shifted);
                z_lam[f] = ac.feasible_lamb[idx];
                z_kap[f] = ac.feasible_kapp[idx];
            }

            double primal_sq = 0.0, dual_sq = 0.0;
            for (Face f : mesh.faces()) {
                double kap_f = avg_kap_pf(f);
                double lam_f = lambda_pf_s[f];
                double r_lam = lam_f - z_lam[f];
                double r_kap = kap_f - z_kap[f];
                mu_lam[f] += rho * r_lam;
                mu_kap[f] += rho * r_kap;
                primal_sq += r_lam * r_lam + r_kap * r_kap;
                double dz_lam = z_lam[f] - z_lam_old[f];
                double dz_kap = z_kap[f] - z_kap_old[f];
                dual_sq   += dz_lam * dz_lam + dz_kap * dz_kap;
            }
            const double r_primal = std::sqrt(primal_sq);
            const double r_dual   = rho * std::sqrt(dual_sq);

            spdlog::info("Patch {} ADMM iter {} : dist_kap={:.6e}, dist_lam={:.6e}, r_primal={:.6e}, r_dual={:.6e}, rho={:.4f}",
                         pd.idx, outer, dist_after_kap, dist_after_lam, r_primal, r_dual, rho);

            if (r_primal < tol_primal && r_dual < tol_dual) {
                spdlog::info("Patch {} ADMM converged at outer iter {}.", pd.idx, outer);
                break;
            }

            if (r_primal > rho_ratio * r_dual && rho < rho_max) {
                rho = std::min(rho_max, rho * rho_growth);
                spdlog::info("  rho up -> {:.4f}", rho);
            } else if (r_dual > rho_ratio * r_primal) {
                rho = std::max(1e-6, rho / rho_growth);
                spdlog::info("  rho down -> {:.4f}", rho);
            }
        }



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
                double sum = 0.0; int cnt = 0;
                for (Vertex v : f.adjacentVertices()) { sum += kappa_pv_s[v]; cnt++; }
                double kap = sum / cnt;
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

