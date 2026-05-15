// Inverse: inverse design on a SINGLE mesh.
//
// Reads:
//   PathSetting.MeshesDir + {model}.obj                  (target geometry)
//   PathSetting.ParamDir  + {model}/{model}_P.obj        (parameterisation)
//   PathSetting.ParamDir  + {model}/global_scale.txt     (uniform scale)
//
// Writes (under ../outputs/{model}/):
//   {model}_targ.obj   — scaled target / globalScale (i.e. unscaled target)
//   {model}_inv.obj    — recovered deformed mesh (Vr / globalScale)
//
// Run Param first to populate PathSetting.ParamDir.

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

int main(int /*argc*/, char* /*argv*/[])
{
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    Config config("cfg.json");
    ActiveComposite ac(config.materialJsonPath());
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    spdlog::info("Inverse (single mesh): start.");

    const std::string model    = config.ModelSetting.ModelName;
    const std::string in_path  = config.PathSetting.MeshesDir + model + config.ModelSetting.Postfix;
    const std::string param_dir = config.PathSetting.ParamDir + model + "/";
    const std::string p_path   = param_dir + model + "_P.obj";
    const std::string sc_path  = param_dir + "global_scale.txt";
    const std::string out_dir    = "../outputs/" + model + "/";
    const std::string design_dir = config.PathSetting.DesignDir + model + "/";
    std::filesystem::create_directories(out_dir);
    std::filesystem::create_directories(design_dir);

    // -------- Load target V --------
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::readOBJ(in_path, V, F)) {
        spdlog::error("Cannot read target mesh: {}", in_path);
        return -1;
    }
    size_t nV = V.rows();
    size_t nF = F.rows();
    spdlog::info("Target mesh: {} V, {} F (from {}).", nV, nF, in_path);

    while (nF < config.RuntimeSetting.nFmin) {
        Eigen::MatrixXd tV = V;
        Eigen::MatrixXi tF = F;
        igl::loop(tV, tF, V, F);
        nV = V.rows();
        nF = F.rows();
    }

    // -------- Load P + scale from Param's output --------
    Eigen::MatrixXd P_loaded3;
    Eigen::MatrixXi F_p;
    if (!igl::readOBJ(p_path, P_loaded3, F_p)) {
        spdlog::error("Cannot read parameterisation: {}.  Run Param first.", p_path);
        return -1;
    }
    if (P_loaded3.rows() != (Eigen::Index)nV) {
        spdlog::error("P vertex count ({}) does not match subdivided target V count ({}).",
                      P_loaded3.rows(), nV);
        return -1;
    }
    Eigen::MatrixXd P(nV, 2);
    P = P_loaded3.leftCols(2);

    double globalScale = 1.0;
    {
        std::ifstream ifs(sc_path);
        if (!ifs.is_open()) {
            spdlog::error("Cannot read global_scale.txt: {}.  Run Param first.", sc_path);
            return -1;
        }
        ifs >> globalScale;
    }
    spdlog::info("globalScale = {:.6f}", globalScale);

    // -------- Apply globalScale to V (P is already scaled by Param) --------
    V *= globalScale;

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

    std::vector<int> fixedVertexIdx = findCenterVertexIndices(P, F);
    std::vector<int> fixedIdx = findCenterFaceIndices(P, F);

    double E = 1.0;
    double nu = 0.5;
    Morphmesh morph_mesh(V, P, F, E, nu);
    Morphmesh::ComputeMorphophing(geometry, V, F, nV, nF, ref_faces,
        MrInv, morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t, morph_mesh.kappa_pv_t, morph_mesh.kappa_pf_t, &morph_mesh.vertex_area_sum);
    Morphmesh::SetMorphophing(morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t,
        morph_mesh.kappa_pv_t,morph_mesh.kappa_pf_t,
        morph_mesh.lambda_pv_s, morph_mesh.lambda_pf_s,
        morph_mesh.kappa_pv_s, morph_mesh.kappa_pf_s);

    auto V_targ = V;
    V_targ *= 1.0 / globalScale;
    igl::writeOBJ(out_dir + model + "_targ.obj", V_targ, F);

    VertexData<double> lambda_pv_s(mesh, morph_mesh.lambda_pv_s);
    VertexData<double> kappa_pv_s(mesh, morph_mesh.kappa_pv_s);
    FaceData<double> lambda_pf_s(mesh, morph_mesh.lambda_pf_s);
    FaceData<double> kappa_pf_s(mesh, morph_mesh.kappa_pf_s);

    auto Vr = V;

    spdlog::info("Step 4: Inverse Design.");

    double wP_kap = config.RuntimeSetting.wP_kap;
    double wP_lam = config.RuntimeSetting.wP_lam;
    double penalty_threshold = config.RuntimeSetting.penalty_threshold;
    double betaP = config.RuntimeSetting.betaP;
    auto penalty_to_lamb = MaterialPenaltyFunctionPerF(geometry, ac.feasible_lamb, betaP);
    auto penalty_to_kapp = MaterialPenaltyFunctionPerV(geometry, ac.feasible_kapp, betaP);

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
    while(k < stage_iter)
    {
        spdlog::info("Stage {}, OptKap start, wP_kap: {:.6f}, wP_lam: {:.6f}.", k, wP_kap, wP_lam);
        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Vr = sparse_gauss_newton_FixLam_OptKap_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s, adjointFunc_OptKap, penalty_to_kapp, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_kap, wL_kap, wP_kap,
            E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces);

        distance = (Vr - targetV).squaredNorm() / nV;
        penalty_kap = compute_candidate_diff(ac.feasible_kapp,kappa_pv_s.toVector(),true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb,lambda_pf_s.toVector(),true);
        spdlog::info("Stage {}, OptKap finish - Distance: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                     k, distance, penalty_kap, penalty_lam);

        spdlog::info("Stage {}, OptLam start, wP_kap: {:.6f}, wP_lam: {:.6f}.", k, wP_kap, wP_lam);
        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pv_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Vr = sparse_gauss_newton_FixKap_OptLam_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s, adjointFunc_OptLam, penalty_to_lamb, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_lam, wL_lam, wP_lam,
            E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces);

        distance = (Vr - targetV).squaredNorm() / nV;
        penalty_kap = compute_candidate_diff(ac.feasible_kapp,kappa_pv_s.toVector(),true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb,lambda_pf_s.toVector(),true);
        spdlog::info("Stage {}, OptLam finish- Distance: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                     k, distance, penalty_kap, penalty_lam);

        k++;
        if (penalty_kap >= penalty_threshold) wP_kap *= 10;
        if (penalty_lam >= penalty_threshold) wP_lam *= 10;
        if (penalty_kap < penalty_threshold && penalty_lam < penalty_threshold) break;
    }

#else

    Vr = targetV;
    while(k < stage_iter)
    {
        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Vr = sparse_gauss_newton_FixLam_OptKap(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s, adjointFunc_OptKap, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, config.RuntimeSetting.wM, config.RuntimeSetting.wL,
            E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces);

        double distance_kap = (Vr - targetV).squaredNorm() / nV;
        spdlog::info("Stage {}, OptKap finish - Distance: {:.6f}", k, distance_kap);

        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pv_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
        Vr = sparse_gauss_newton_FixKap_OptLam(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s, adjointFunc_OptLam, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, 0.0, config.RuntimeSetting.wL,
            E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces);

        double distance_lam = (Vr - targetV).squaredNorm() / nV;
        spdlog::info("Stage {}, OptLam finish - Distance: {:.6f}", k, distance_lam);

        k++;
    }

#endif

    auto V_inv = Vr;
    V_inv *= 1.0 / globalScale;
    igl::writeOBJ(out_dir + model + "_inv.obj", V_inv, F);

    // ---- Material projection: per-face (lambda, kappa) -> nearest feasible (t1, t2) ----
    {
        const std::string mat_path = design_dir + model + "_material.txt";
        std::ofstream ofs(mat_path);
        ofs << "# face_id  t1  t2\n";
        for (Face f : mesh.faces()) {
            double sum = 0.0; int cnt = 0;
            for (Vertex v : f.adjacentVertices()) { sum += kappa_pv_s[v]; cnt++; }
            double kap = sum / cnt;
            double lam = lambda_pf_s[f];
            int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap, lam);
            double t1 = ac.feasible_t_vals[idx].first;
            double t2 = ac.feasible_t_vals[idx].second;
            ofs << f.getIndex() << "  " << t1 << "  " << t2 << "\n";
        }
        spdlog::info("Material -> {}", mat_path);
    }

    spdlog::info("Inverse (single mesh): done.  Output → {}", out_dir);
    return 0;
}
