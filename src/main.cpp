// IglViewer.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

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

// #define __VERFIY_FORWARD_PREDIT__
#define __VERFIY_INVERSE_DESIGN__

#define __Add_PENALTY__

int main(int argc, char* argv[])
{


    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    Config config("cfg.json");
    ActiveComposite ac(config.ResourceSetting.MaterialPath);
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    spdlog::info("program start:");

	///***************************************** Load Mesh *****************************************///

    std::string input_mesh_path = config.ModelSetting.InputPath +  config.ModelSetting.ModelName + config.ModelSetting.Postfix;
    if (!igl::readOBJ(input_mesh_path, V, F)) {
        spdlog::error("Error: Could not read output_mesh.obj\n");
        return -1;
    }

    size_t nV = V.rows();
    size_t nF = F.rows();
    spdlog::info("Read meshes with {0} vertices and {1} faces.",nV,nF);


    while (nF < config.RuntimeSetting.nFmin)
    {
        Eigen::MatrixXd tempV = V;
        Eigen::MatrixXi tempF = F;
        igl::loop(tempV, tempF, V, F);
        nV = V.rows();
        nF = F.rows();
    }

    double scaleFactor = 1.0;
    double scaleFactor_1 = config.RuntimeSetting.Platewidth / (V.colwise().maxCoeff() - V.colwise().minCoeff()).maxCoeff();
    scaleFactor *= scaleFactor_1;
    V *= scaleFactor_1;

    ManifoldSurfaceMesh mesh(F);
    VertexPositionGeometry geometry(mesh, V);
    geometry.refreshQuantities();

    ///***************************************** Parameterization *****************************************///
    spdlog::info("Step 1: Parameterization.");
    Eigen::MatrixXd P = parameterization(V, F, ac.range_lam.x, ac.range_lam.y, 0);


    double scaleFactor_2 = config.RuntimeSetting.Platewidth / (P.colwise().maxCoeff() - P.colwise().minCoeff()).maxCoeff();
    scaleFactor *= scaleFactor_2;
    V *= scaleFactor_2;
    P *= scaleFactor_2;
    geometry.inputVertexPositions *= scaleFactor_2;
    geometry.refreshQuantities();


    std::vector<bool> boundary_vertex_flags(nV, false);
    std::vector<bool> boundary_face_flags(nF, false);
    std::vector<int> boundary_ref_indices(nF, 0);


    ///***************************************** Material Settings *****************************************///

    spdlog::info("Step 2: Material Settings.");

    Eigen::MatrixXd targetV = V;

    FaceData<Eigen::MatrixXd> M = precomputeM(mesh, V, F);
    FaceData<Eigen::Matrix2d> MrInv = precomputeMrInv(mesh, P, F);

    std::vector<int> fixedVertexIdx = findCenterVertexIndices(P, F);
    std::vector<int> fixedVertexIdx_x = {};
    std::vector<int> fixedVertexIdx_y = {};
    std::vector<int> fixedVertexIdx_z = {};
    std::vector<int> fixedIdx;

    double E = 1.0;
    double nu = 0.5;
    Morphmesh morph_mesh(V, P, F, E, nu);
    Morphmesh::ComputeMorphophing(geometry, V, F, nV, nF,boundary_vertex_flags,boundary_face_flags,boundary_ref_indices,
        MrInv, morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t,morph_mesh.kappa_pv_t,morph_mesh.kappa_pf_t, &morph_mesh.vertex_area_sum);
    Morphmesh::SetMorphophing(morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t,
        morph_mesh.kappa_pv_t,morph_mesh.kappa_pf_t,
        morph_mesh.lambda_pv_s, morph_mesh.lambda_pf_s,
        morph_mesh.kappa_pv_s, morph_mesh.kappa_pf_s);
	// Morphmesh::RestrictRange(morph_mesh.lambda_pv_s, ac.range_lam.x, ac.range_lam.y);
	// Morphmesh::RestrictRange(morph_mesh.kappa_pv_s, ac.range_kap.x, ac.range_kap.y);
    // Morphmesh::RestrictRange(morph_mesh.lambda_pf_s, ac.range_lam.x, ac.range_lam.y);
    // Morphmesh::RestrictRange(morph_mesh.kappa_pf_s, ac.range_kap.x, ac.range_kap.y);
 

    auto V_targ= V;
    V_targ *= 1.0 / scaleFactor;
    std::string output_mesh_targ_path = config.OutputSetting.OutputPath +
        config.ModelSetting.ModelName + "_targ" + ".obj";
    igl::writeOBJ(output_mesh_targ_path, V_targ, F);


    FaceData<bool> is_boundary_face(mesh);
    FaceData<int> boundary_ref_index(mesh);
    VertexData<bool> is_boundary_vertex(mesh);
    VertexData<double> lambda_pv_s(mesh, morph_mesh.lambda_pv_s);
    VertexData<double> kappa_pv_s(mesh, morph_mesh.kappa_pv_s);
    FaceData<double> lambda_pf_s(mesh, morph_mesh.lambda_pf_s);
    FaceData<double> kappa_pf_s(mesh, morph_mesh.kappa_pf_s);

    ///***************************************** Forward Predit *****************************************///

    // spdlog::info("Step 3: Forward Predit.");


    auto V_pred = V, Vr = V;

    // auto simul_func_2 = simulationFunction(geometry,
    //     MrInv, 
    //     lambda_pv_s,
    //     kappa_pv_s, 
	// 	E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b
    //     );

    // Vr = V;
    // newton(geometry, Vr, simul_func_2, config.RuntimeSetting.MaxIter, 
    //     config.RuntimeSetting.epsilon, true, fixedIdx);

    // V_pred = Vr;
    // V_pred *= 1.0 / scaleFactor;
    // std::string output_mesh_pred_path_2 = config.OutputSetting.OutputPath + 
    //     config.ModelSetting.ModelName + "_pred" + ".obj";
    // igl::writeOBJ(output_mesh_pred_path_2, V_pred, F);


    // ///***************************************** Inverse Design *****************************************///

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
        spdlog::info("Stage {}, OptKap start, wP_kap: {:.6f}, wP_lam: {:.6f}.", k, wP_kap, wP_lam);
        // Vr = targetV;
        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
        Vr = sparse_gauss_newton_FixLam_OptKap_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s, adjointFunc_OptKap, penalty_to_kapp, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_kap, wL_kap, wP_kap,
            E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b);

        // Compute and output distance and penalties after OptKap
        distance = (Vr - targetV).squaredNorm() / nV;
        penalty_kap = compute_candidate_diff(ac.feasible_kapp,kappa_pv_s.toVector(),true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb,lambda_pf_s.toVector(),true);
        spdlog::info("Stage {}, OptKap finish - Distance: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                     k, distance, penalty_kap, penalty_lam);



        spdlog::info("Stage {}, OptLam start, wP_kap: {:.6f}, wP_lam: {:.6f}.", k, wP_kap, wP_lam);
        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pv_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
        Vr = sparse_gauss_newton_FixKap_OptLam_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s, adjointFunc_OptLam, penalty_to_lamb, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_lam, wL_lam, wP_lam,
            E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b);

        // Compute and output distance and penalties after OptLam
        distance = (Vr - targetV).squaredNorm() / nV;
        penalty_kap = compute_candidate_diff(ac.feasible_kapp,kappa_pv_s.toVector(),true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb,lambda_pf_s.toVector(),true);
        spdlog::info("Stage {}, OptLam finish- Distance: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                     k, distance, penalty_kap, penalty_lam);

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
                E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
            Eigen::MatrixXd Vr_proj = Vr;
            newton(geometry, Vr_proj, simFunc_proj,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx);
            double dist_proj = (Vr_proj - targetV).squaredNorm() / nV;

            spdlog::info("Stage {}, Projected distance: {:.6f}", k, dist_proj);
        }

        k++;
        if (penalty_kap >= penalty_threshold) {
            wP_kap *= 10;
        }
        if (penalty_lam >= penalty_threshold) {
            wP_lam *= 10;
        }

        wM_kap *= 0.1;
        wL_kap *= 0.1;
        wM_lam *= 0.1;
        wL_lam *= 0.1;

        if(penalty_kap < penalty_threshold && penalty_lam < penalty_threshold)
            break;

        // wM_kap *=0.1;
        // wM_lam *=0.1;
        // wL_kap *=0.1;
        // wL_lam *=0.1;
    }


#else

    Vr = targetV;
    while(k < stage_iter)
    {
        spdlog::info("Stage {}, OptKap start", k);
        
        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
        Vr = sparse_gauss_newton_FixLam_OptKap(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s, adjointFunc_OptKap, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, config.RuntimeSetting.wM, config.RuntimeSetting.wL,
            E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b);

        // Compute and output distance after OptKap
        double distance_kap = (Vr - targetV).squaredNorm() / nV;
        spdlog::info("Stage {}, OptKap finish - Distance: {:.6f}", k, distance_kap);


        spdlog::info("Stage {}, OptLam start", k);
        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pv_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
        Vr = sparse_gauss_newton_FixKap_OptLam(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s, adjointFunc_OptLam, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, 0.0, config.RuntimeSetting.wL,
            E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b);

        // morph_mesh.lambda_pv_s = lambda_pv_s.toVector();
        // morph_mesh.kappa_pv_s = kappa_pv_s.toVector();

        // // Reassign per-face values from optimized per-vertex values
        // Morphmesh::ReassignPFFromPV(geometry, F, morph_mesh.lambda_pv_s, morph_mesh.kappa_pv_s,
        //     morph_mesh.lambda_pf_s, morph_mesh.kappa_pf_s);
        // lambda_pf_s = FaceData<double>(mesh, morph_mesh.lambda_pf_s);
        // kappa_pf_s = FaceData<double>(mesh, morph_mesh.kappa_pf_s);

        // Compute and output distance after OptLam
        double distance_lam = (Vr - targetV).squaredNorm() / nV;
        spdlog::info("Stage {}, OptLam finish - Distance: {:.6f}", k, distance_lam);


        k++;
    }

    
#endif



    auto V_inv = Vr;
    V_inv *= 1.0 / scaleFactor;
    std::string output_mesh_inv_path = config.OutputSetting.OutputPath +
        config.ModelSetting.ModelName + "_inv" + ".obj";
    igl::writeOBJ(output_mesh_inv_path, V_inv, F);



    ///***************************************** View by Imgui *****************************************///               
    //igl::opengl::glfw::Viewer viewer;
    //viewer.data().set_mesh(V_pred, F);
    ////viewer.data().set_colors(C);
    //viewer.data().show_lines = true;
    //viewer.launch();

    spdlog::info("program finish.");

}

