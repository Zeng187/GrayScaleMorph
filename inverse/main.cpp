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
                              morph_mesh.kappa_pv_s, morph_mesh.kappa_pf_s);

    // Mirror the target into outputs/ so {targ, inv} sit side-by-side for
    // easy comparison.  Content is the same as 2_target/.../patch_0_V.obj.
    igl::writeOBJ(morph_dir + "patch_0_targ.obj", V, F);

    VertexData<double> lambda_pv_s(mesh, morph_mesh.lambda_pv_s);
    VertexData<double> kappa_pv_s(mesh, morph_mesh.kappa_pv_s);
    FaceData<double> lambda_pf_s(mesh, morph_mesh.lambda_pf_s);
    FaceData<double> kappa_pf_s(mesh, morph_mesh.kappa_pf_s);

    auto Vr = V;

    spdlog::info("Step 4: Inverse Design (ADMM + Augmented Lagrangian).");

    const double wM_kap = config.RuntimeSetting.wM_kap;
    const double wM_lam = config.RuntimeSetting.wM_lam;
    const double wL_kap = config.RuntimeSetting.wL_kap;
    const double wL_lam = config.RuntimeSetting.wL_lam;

    // ADMM hyperparameters
    double       rho            = config.RuntimeSetting.rho;
    const double rho_max        = config.RuntimeSetting.rho_max;
    const double rho_growth     = config.RuntimeSetting.rho_growth;
    const double rho_ratio      = config.RuntimeSetting.rho_ratio;
    const int    max_outer_iter = config.RuntimeSetting.max_outer_iter;
    const double tol_primal     = config.RuntimeSetting.tol_primal;
    const double tol_dual       = config.RuntimeSetting.tol_dual;

    // ADMM state — per-face (z = discrete projection target, mu = Lagrange multiplier).
    FaceData<double> z_lam(mesh, 0.0), z_kap(mesh, 0.0);
    FaceData<double> mu_lam(mesh, 0.0), mu_kap(mesh, 0.0);

    // Helper: average per-vertex κ to per-face (1/3 sum).
    auto avg_kap_pf = [&](Face f) {
        double sum = 0.0; int cnt = 0;
        for (Vertex v : f.adjacentVertices()) { sum += kappa_pv_s[v]; cnt++; }
        return sum / cnt;
    };

    // Initial z = Proj_F( current (λ_pf, κ_pf_avg) )  — closest feasible per face.
    for (Face f : mesh.faces()) {
        double kap_f = avg_kap_pf(f);
        double lam_f = lambda_pf_s[f];
        int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap_f, lam_f);
        z_lam[f] = ac.feasible_lamb[idx];
        z_kap[f] = ac.feasible_kapp[idx];
    }

    for (int outer = 0; outer < max_outer_iter; ++outer)
    {
        spdlog::info("ADMM iter {} start  (rho = {:.4f})", outer, rho);

        // ---- Step 1a: θ-update, OptKap (κ_pv at face-level AL coupling) ----
        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Vr = sparse_gauss_newton_FixLam_OptKap_AL(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s,
            adjointFunc_OptKap, z_kap, mu_kap, rho, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
            wM_kap, wL_kap, E, nu, ac.thickness,
            config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        double dist_after_kap = (Vr - targetV).squaredNorm() / nV;

        // ---- Step 1b: θ-update, OptLam (λ_pf directly face-level) ----
        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pv_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Vr = sparse_gauss_newton_FixKap_OptLam_AL(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s,
            adjointFunc_OptLam, z_lam, mu_lam, rho, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
            wM_lam, wL_lam, E, nu, ac.thickness,
            config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        double dist_after_lam = (Vr - targetV).squaredNorm() / nV;

        // ---- Step 2: z-update.  z_f = Proj_F(θ_f + μ_f / ρ),per face,joint (λ, κ) ----
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

        // ---- Step 3: dual ascent  μ ← μ + ρ·(θ - z) ----
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

        // Projected distance: build a per-face (λ, κ) field from z and evaluate ||Vr - target|| on it
        // (κ broadcast to vertices for use with simulationFunction in OptLam variant — diagnostic only).
        double dist_proj = std::numeric_limits<double>::quiet_NaN();
        spdlog::info("ADMM iter {} : dist_kap={:.6e}, dist_lam={:.6e}, r_primal={:.6e}, r_dual={:.6e}, rho={:.4f}",
                     outer, dist_after_kap, dist_after_lam, r_primal, r_dual, rho);

        // Early stop on primal & dual residual
        if (r_primal < tol_primal && r_dual < tol_dual) {
            spdlog::info("ADMM converged at outer iter {} (primal {:.3e}, dual {:.3e}).",
                         outer, r_primal, r_dual);
            break;
        }

        // Adaptive ρ (Boyd 2011 §3.4.1)
        if (r_primal > rho_ratio * r_dual && rho < rho_max) {
            rho = std::min(rho_max, rho * rho_growth);
            spdlog::info("  rho up -> {:.4f}", rho);
        } else if (r_dual > rho_ratio * r_primal) {
            rho = std::max(1e-6, rho / rho_growth);
            spdlog::info("  rho down -> {:.4f}", rho);
        }
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
        const std::string lk_path  = design_dir + "patch_0_lamkap.txt";
        std::ofstream mof(mat_path);
        std::ofstream lof(lk_path);
        mof << "# face_id  t1  t2\n";
        lof << "# face_id  lambda  kappa\n";
        for (Face f : mesh.faces())
        {
            double sum = 0.0;
            int cnt = 0;
            for (Vertex v : f.adjacentVertices())
            {
                sum += kappa_pv_s[v];
                cnt++;
            }
            double kap = sum / cnt;
            double lam = lambda_pf_s[f];
            int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap, lam);
            double t1 = ac.feasible_t_vals[idx].first;
            double t2 = ac.feasible_t_vals[idx].second;
            mof << f.getIndex() << "  " << t1  << "  " << t2  << "\n";
            lof << f.getIndex() << "  " << lam << "  " << kap << "\n";
        }
        spdlog::info("Material -> {}", mat_path);
        spdlog::info("LamKap   -> {}", lk_path);
    }

    spdlog::info("Inverse (single mesh): done.  Output -> {}", morph_dir);
    return 0;
}
