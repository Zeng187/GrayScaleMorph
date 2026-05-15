

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

    const std::string model = config.ModelSetting.ModelName;
    const std::string patches_dir = config.PathSetting.SegmentDir + model + "_original_planA/patches/";
    const std::string out_dir     = "../outputs/" + model + "/";
    const std::string design_dir  = config.PathSetting.DesignDir + model + "_original_planA/";
    std::filesystem::create_directories(out_dir);
    std::filesystem::create_directories(design_dir);
    spdlog::info("Patches dir : {}", patches_dir);
    spdlog::info("Output dir  : {}", out_dir);
    spdlog::info("Design dir  : {}", design_dir);

    ///***************************************** Phase 1: read + parameterize each patch *****************************************///

    struct PatchData {
        size_t idx;
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        Eigen::MatrixXd P;
        size_t nV, nF;
    };
    std::vector<PatchData> patches;

    for (size_t i = 0;; ++i) {
        std::string path = patches_dir + "patch_" + std::to_string(i) + ".obj";
        Eigen::MatrixXd Vp;
        Eigen::MatrixXi Fp;
        if (!igl::readOBJ(path, Vp, Fp)) {
            spdlog::info("No more patches after idx {}.", i - 1);
            break;
        }
        size_t nV = Vp.rows();
        size_t nF = Fp.rows();
        spdlog::info("Patch {}: read {} V, {} F.", i, nV, nF);

        // Loop subdivision until nF reaches nFmin
        while (nF < config.RuntimeSetting.nFmin) {
            Eigen::MatrixXd tV = Vp;
            Eigen::MatrixXi tF = Fp;
            igl::loop(tV, tF, Vp, Fp);
            nV = Vp.rows();
            nF = Fp.rows();
        }
        if (nV != Vp.rows() || nF != Fp.rows()) {
            spdlog::info("Patch {}: subdivided to {} V, {} F.", i, nV, nF);
        }

        // Parameterize (gauge shift is applied internally on P; V untouched)
        spdlog::info("Patch {}: parameterize ...", i);
        Eigen::MatrixXd P = parameterization(Vp, Fp, ac.range_lam.x, ac.range_lam.y, 0);

        PatchData pd;
        pd.idx = i;
        pd.V = std::move(Vp);
        pd.F = std::move(Fp);
        pd.P = std::move(P);
        pd.nV = nV;
        pd.nF = nF;
        patches.push_back(std::move(pd));
    }
    spdlog::info("Total patches: {}", patches.size());

    if (patches.empty()) {
        spdlog::error("No patches found, abort.");
        return -1;
    }

    ///***************************************** Phase 2: globalScale = min(platewidth / P_extent_i) *****************************************///

    double globalScale = std::numeric_limits<double>::infinity();
    for (const auto& pd : patches) {
        const double P_extent = (pd.P.colwise().maxCoeff() - pd.P.colwise().minCoeff()).maxCoeff();
        const double scale_i = config.RuntimeSetting.Platewidth / P_extent;
        spdlog::info("Patch {}: P_extent = {:.4f}, scale_i = {:.6f}", pd.idx, P_extent, scale_i);
        if (scale_i < globalScale) globalScale = scale_i;
    }
    spdlog::info("globalScale (shared across all patches) = {:.6f}", globalScale);

    ///***************************************** Phase 3: per-patch inverse design *****************************************///

    for (auto& pd : patches) {
        spdlog::info("=================================================");
        spdlog::info("Inverse design for patch {} ({} V, {} F)", pd.idx, pd.nV, pd.nF);
        spdlog::info("=================================================");

        // Uniformly scale V and P by globalScale -> per-face lambda is invariant
        // under such uniform scaling, so gauge-shifted P stays centered in the
        // material window even after this scaling.
        pd.V *= globalScale;
        pd.P *= globalScale;

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

        std::vector<int> fixedVertexIdx = findCenterVertexIndices(P, F);
        std::vector<int> fixedIdx = findCenterFaceIndices(P, F);  // 9 DOF indices

        double E = 1.0;
        double nu = 0.5;
        Morphmesh morph_mesh(V, P, F, E, nu);
        Morphmesh::ComputeMorphophing(geometry, V, F, nV, nF, ref_faces,
            MrInv, morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t, morph_mesh.kappa_pv_t, morph_mesh.kappa_pf_t, &morph_mesh.vertex_area_sum);
        Morphmesh::SetMorphophing(morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t,
            morph_mesh.kappa_pv_t,morph_mesh.kappa_pf_t,
            morph_mesh.lambda_pv_s, morph_mesh.lambda_pf_s,
            morph_mesh.kappa_pv_s, morph_mesh.kappa_pf_s);

        // Write target obj (restored to original (pre-scale) coordinates)
        auto V_targ = V;
        V_targ *= 1.0 / globalScale;
        std::string output_mesh_targ_path = out_dir + "patch_" + std::to_string(pd.idx) + "_targ.obj";
        igl::writeOBJ(output_mesh_targ_path, V_targ, F);


        VertexData<double> lambda_pv_s(mesh, morph_mesh.lambda_pv_s);
        VertexData<double> kappa_pv_s(mesh, morph_mesh.kappa_pv_s);
        FaceData<double> lambda_pf_s(mesh, morph_mesh.lambda_pf_s);
        FaceData<double> kappa_pf_s(mesh, morph_mesh.kappa_pf_s);

        ///***************************************** Inverse Design *****************************************///

        auto V_pred = V, Vr = V;

        spdlog::info("Step 4: Inverse Design.");

        double wP_kap = config.RuntimeSetting.wP_kap;
        double wP_lam = config.RuntimeSetting.wP_lam;
        double penalty_threshold = config.RuntimeSetting.penalty_threshold;
        double betaP = config.RuntimeSetting.betaP;
        auto penalty_to_lamb = MaterialPenaltyFunctionPerF(geometry, ac.feasible_lamb, betaP);
        auto penalty_to_kapp = MaterialPenaltyFunctionPerV(geometry, ac.feasible_kapp, betaP);
        auto penalty_to_modu = MaterialPenaltyFunctionPerV(geometry, ac.feasible_modl, betaP);

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
            spdlog::info("Patch {} Stage {}, OptKap start, wP_kap: {:.6f}, wP_lam: {:.6f}.", pd.idx, k, wP_kap, wP_lam);
            auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixLam_OptKap_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s, adjointFunc_OptKap, penalty_to_kapp, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_kap, wL_kap, wP_kap,
                E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces);

            distance = (Vr - targetV).squaredNorm() / nV;
            penalty_kap = compute_candidate_diff(ac.feasible_kapp,kappa_pv_s.toVector(),true);
            penalty_lam = compute_candidate_diff(ac.feasible_lamb,lambda_pf_s.toVector(),true);
            spdlog::info("Patch {} Stage {}, OptKap finish - Distance: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                         pd.idx, k, distance, penalty_kap, penalty_lam);



            spdlog::info("Patch {} Stage {}, OptLam start, wP_kap: {:.6f}, wP_lam: {:.6f}.", pd.idx, k, wP_kap, wP_lam);
            auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pv_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixKap_OptLam_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s, adjointFunc_OptLam, penalty_to_lamb, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_lam, wL_lam, wP_lam,
                E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces);

            distance = (Vr - targetV).squaredNorm() / nV;
            penalty_kap = compute_candidate_diff(ac.feasible_kapp,kappa_pv_s.toVector(),true);
            penalty_lam = compute_candidate_diff(ac.feasible_lamb,lambda_pf_s.toVector(),true);
            spdlog::info("Patch {} Stage {}, OptLam finish- Distance: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                         pd.idx, k, distance, penalty_kap, penalty_lam);

            // Evaluate distance after jointly projecting kappa and lambda to the same feasible index
            {
                FaceData<double> kappa_pf_proj(mesh);
                FaceData<double> lambda_pf_proj(mesh);

                for (Face f : mesh.faces()) {
                    double sum = 0.0; int cnt = 0;
                    for (Vertex v : f.adjacentVertices()) { sum += kappa_pv_s[v]; cnt++; }
                    double kap = sum / cnt;
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
        while(k < stage_iter)
        {
            spdlog::info("Patch {} Stage {}, OptKap start", pd.idx, k);

            auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixLam_OptKap(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s, adjointFunc_OptKap, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, config.RuntimeSetting.wM, config.RuntimeSetting.wL,
                E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces);

            double distance_kap = (Vr - targetV).squaredNorm() / nV;
            spdlog::info("Patch {} Stage {}, OptKap finish - Distance: {:.6f}", pd.idx, k, distance_kap);


            spdlog::info("Patch {} Stage {}, OptLam start", pd.idx, k);
            auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pv_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixKap_OptLam(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s, adjointFunc_OptLam, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, 0.0, config.RuntimeSetting.wL,
                E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces);

            double distance_lam = (Vr - targetV).squaredNorm() / nV;
            spdlog::info("Patch {} Stage {}, OptLam finish - Distance: {:.6f}", pd.idx, k, distance_lam);


            k++;
        }


#endif



        auto V_inv = Vr;
        V_inv *= 1.0 / globalScale;
        std::string output_mesh_inv_path = out_dir + "patch_" + std::to_string(pd.idx) + "_inv.obj";
        igl::writeOBJ(output_mesh_inv_path, V_inv, F);

        // ---- Material projection: per-face (lambda, kappa) -> nearest feasible (t1, t2) ----
        {
            std::string mat_path = design_dir + "patch_" + std::to_string(pd.idx) + "_material.txt";
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
            spdlog::info("Patch {} material -> {}", pd.idx, mat_path);
        }

        spdlog::info("Patch {} done.", pd.idx);
    }

    spdlog::info("All patches processed; program finish.");

}

