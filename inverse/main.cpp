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
                              morph_mesh.kappa_pf_s, morph_mesh.kappa_pf_s);

    // Mirror the target into outputs/ so {targ, inv} sit side-by-side for
    // easy comparison.  Content is the same as 2_target/.../patch_0_V.obj.
    igl::writeOBJ(morph_dir + "patch_0_targ.obj", V, F);

    VertexData<double> lambda_pv_s(mesh, morph_mesh.lambda_pv_s);
    FaceData<double> lambda_pf_s(mesh, morph_mesh.lambda_pf_s);
    FaceData<double> kappa_pf_s(mesh, morph_mesh.kappa_pf_s);

    auto Vr = V;
    auto V_init = V;
    // V_init.col(0) = P.col(0);
    // V_init.col(1) = P.col(1);
    // V_init.col(2).setConstant(0.0);
    Vr = V_init;
    


    spdlog::info("Step 4: Inverse Design.");

    double wP_kap = config.RuntimeSetting.wP_kap;
    double wP_lam = config.RuntimeSetting.wP_lam;
    double penalty_threshold = config.RuntimeSetting.penalty_threshold;
    double betaP = config.RuntimeSetting.betaP;
    auto penalty_to_lamb = MaterialPenaltyFunctionPerF(geometry, ac.feasible_lamb, betaP);
    auto penalty_to_kapp = MaterialPenaltyFunctionPerF(geometry, ac.feasible_kapp, betaP);

    int stage_iter = 5;
    int k = 0;

    double wM_kap = config.RuntimeSetting.wM_kap;
    double wM_lam = config.RuntimeSetting.wM_lam;
    double wL_kap = config.RuntimeSetting.wL_kap;
    double wL_lam = config.RuntimeSetting.wL_lam;

#ifdef __Add_PENALTY__

    double distance = 0.0;
    double penalty_kap = 0.0;
    double penalty_lam = 0.0;

    while (k < stage_iter)
    {
        spdlog::info("Stage {}, OptKap start, wP_kap: {:.6f}, wP_lam: {:.6f}.", k, wP_kap, wP_lam);
        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Vr = sparse_gauss_newton_FixLam_OptKap_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, adjointFunc_OptKap, penalty_to_kapp, fixedIdx,
                                                       config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_kap, wL_kap, wP_kap,
                                                       E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);

        distance = (Vr - targetV).squaredNorm() / nV;
        penalty_kap = compute_candidate_diff(ac.feasible_kapp, kappa_pf_s.toVector(), true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
        spdlog::info("Stage {}, OptKap finish - Distance: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                     k, distance, penalty_kap, penalty_lam);

        spdlog::info("Stage {}, OptLam start, wP_kap: {:.6f}, wP_lam: {:.6f}.", k, wP_kap, wP_lam);
        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Vr = sparse_gauss_newton_FixKap_OptLam_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, adjointFunc_OptLam, penalty_to_lamb, fixedIdx,
                                                       config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_lam, wL_lam, wP_lam,
                                                       E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);

        distance = (Vr - targetV).squaredNorm() / nV;
        penalty_kap = compute_candidate_diff(ac.feasible_kapp, kappa_pf_s.toVector(), true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
        spdlog::info("Stage {}, OptLam finish- Distance: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                     k, distance, penalty_kap, penalty_lam);

        k++;
        if (penalty_kap >= penalty_threshold)
            wP_kap *= 10;
        if (penalty_lam >= penalty_threshold)
            wP_lam *= 10;
        if (penalty_kap < penalty_threshold && penalty_lam < penalty_threshold)
            break;

        wM_kap *= 0.5;
        wL_kap *= 0.5;
        wM_lam *= 0.5;
        wL_lam *= 0.5;
    }

#else

    Vr = targetV;
    while (k < stage_iter)
    {
        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Vr = sparse_gauss_newton_FixLam_OptKap(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, adjointFunc_OptKap, fixedIdx,
                                               config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, config.RuntimeSetting.wM, config.RuntimeSetting.wL,
                                               E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);

        double distance_kap = (Vr - targetV).squaredNorm() / nV;
        spdlog::info("Stage {}, OptKap finish - Distance: {:.6f}", k, distance_kap);

        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Vr = sparse_gauss_newton_FixKap_OptLam(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, adjointFunc_OptLam, fixedIdx,
                                               config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, 0.0, config.RuntimeSetting.wL,
                                               E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);

        double distance_lam = (Vr - targetV).squaredNorm() / nV;
        spdlog::info("Stage {}, OptLam finish - Distance: {:.6f}", k, distance_lam);

        k++;
    }

#endif

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
            double kap = kappa_pf_s[f];
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
