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

int main(int argc, char* argv[])
{


    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    Config config("cfg.json");
    ActiveComposite ac(config.ResourceSetting.MaterialPath);
    ac.ComputeMaterialCurve();

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
        MrInv, morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t,morph_mesh.kappa_pv_t,morph_mesh.kappa_pf_t);
    Morphmesh::SetMorphophing(morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t,
        morph_mesh.kappa_pv_t,morph_mesh.kappa_pf_t,
        morph_mesh.lambda_pv_s, morph_mesh.lambda_pf_s,
        morph_mesh.kappa_pv_s, morph_mesh.kappa_pf_s);
	Morphmesh::RestrictRange(morph_mesh.lambda_pv_s, ac.range_lam.x, ac.range_lam.y);
	Morphmesh::RestrictRange(morph_mesh.kappa_pv_s, ac.range_kap.x, ac.range_kap.y);
    Morphmesh::RestrictRange(morph_mesh.lambda_pf_s, ac.range_lam.x, ac.range_lam.y);
    Morphmesh::RestrictRange(morph_mesh.kappa_pf_s, ac.range_kap.x, ac.range_kap.y);

    // Compute thickness layer values from lambda and kappa
    // Inverts material curve to find t parameters that produce desired lambda/kappa
    Morphmesh::ComputeTLayersFromMorphophing(morph_mesh.lambda_pv_s,morph_mesh.kappa_pv_s,
        ac.m_strain_curve,ac.m_moduls_curve,ac.thickness,
        morph_mesh.t_layer_pv_1,morph_mesh.t_layer_pv_2);

    Morphmesh::ComputeMorphingFormTLayers(morph_mesh.t_layer_pv_1,morph_mesh.t_layer_pv_2,
        ac.m_strain_curve,ac.m_moduls_curve,ac.thickness,
        morph_mesh.lambda_pv_s,morph_mesh.kappa_pv_s);


    ///***************************************** Forward Predit *****************************************///

    spdlog::info("Step 3: Forward Predit.");

    // boundary setting
    FaceData<bool> is_boundary_face(mesh);
    FaceData<int> boundary_ref_index(mesh);
    VertexData<bool> is_boundary_vertex(mesh);

    auto V_pred = V, Vr = V;

    // (1) Simulation from t_layer_1 and t_layer_2
    VertexData<double> t_layer_pv_1_s(mesh,morph_mesh.t_layer_pv_1);
    VertexData<double> t_layer_pv_2_s(mesh,morph_mesh.t_layer_pv_2);

    auto simul_func_1 = simulationFunctionWithMaterial(geometry,
        MrInv, 
        t_layer_pv_1_s,
        t_layer_pv_2_s, 
        ac.m_strain_curve,
        ac.m_moduls_curve,
		E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b
        );

    Vr = V;
    newton(geometry, Vr, simul_func_1, config.RuntimeSetting.MaxIter, 
        config.RuntimeSetting.epsilon, true, fixedIdx);

    V_pred = Vr;
    V_pred *= 1.0 / scaleFactor;
    std::string output_mesh_pred_path_1 = config.OutputSetting.OutputPath + 
        config.ModelSetting.ModelName + "_pred_1" + ".obj";
    igl::writeOBJ(output_mesh_pred_path_1, V_pred, F);

    #ifdef __VERFIY_FORWARD_PREDIT__

    // (2) Simulation from lambda and kappa
    VertexData<double> lambda_pv_s(mesh, morph_mesh.lambda_pv_s);
    VertexData<double> kappa_pv_s(mesh, morph_mesh.kappa_pv_s);


    auto simul_func_2 = simulationFunction(geometry,
        MrInv, 
        lambda_pv_s,
        kappa_pv_s, 
		E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b
        );

    Vr = V;
    newton(geometry, Vr, simul_func_2, config.RuntimeSetting.MaxIter, 
        config.RuntimeSetting.epsilon, true, fixedIdx);

    V_pred = Vr;
    V_pred *= 1.0 / scaleFactor;
    std::string output_mesh_pred_path_2 = config.OutputSetting.OutputPath + 
        config.ModelSetting.ModelName + "_pred_2" + ".obj";
    igl::writeOBJ(output_mesh_pred_path_2, V_pred, F);

    #endif

    Morphmesh::ComputeMorphophing(geometry, V_pred, F, nV,nF,
        boundary_vertex_flags, boundary_face_flags, boundary_ref_indices,
        MrInv, morph_mesh.lambda_pv_r, morph_mesh.lambda_pf_r, morph_mesh.kappa_pv_r,
        morph_mesh.kappa_pf_r);

	std::vector <Eigen::VectorXd> per_vertex_attributes_list{
		morph_mesh.lambda_pv_t, morph_mesh.kappa_pv_t,
        morph_mesh.lambda_pv_s, morph_mesh.kappa_pv_s,
		morph_mesh.lambda_pv_r, morph_mesh.kappa_pv_r,
        morph_mesh.t_layer_pv_1, morph_mesh.t_layer_pv_2
    };
    std::vector <std::string> per_vertex_attributes_str_list{
       "lambda_pv_t", "kappa_pv_t",
       "lambda_pv_s", "kappa_pv_s",
       "lambda_pv_r", "kappa_pv_r",
       "t_layer_pv_1", "t_layer_pv_2"
    };
    std::string output_vtk_pred_path = config.OutputSetting.OutputPath + 
        config.ModelSetting.ModelName + "_pred" + ".vtk";
    write_output_vtk(output_vtk_pred_path, V_pred, F,
                      per_vertex_attributes_list, per_vertex_attributes_str_list);

    // ///***************************************** Inverse Design *****************************************///

    spdlog::info("Step 4: Inverse Design.");

    // Setup vertex data for t_layer_1 (fixed) and t_layer_2 (to be optimized)
    VertexData<double> t_layer_pv_1_vd(mesh, morph_mesh.t_layer_pv_1);
    VertexData<double> t_layer_pv_2_vd(mesh, morph_mesh.t_layer_pv_2);

    // t_layer_pv_1_vd.fill(1.0);
    // t_layer_pv_2_vd.fill(0.0);
    

    // Create adjoint function with material-based computation
    // This will optimize both vertex positions and t_layer_2 values
    auto adjointFunc_Lay1 = adjointFunctionWithMaterial_Lay1(geometry, F, MrInv,
        t_layer_pv_1_vd,
        ac.m_strain_curve,
        ac.m_moduls_curve,
        E, nu, ac.thickness,
        config.RuntimeSetting.w_s,
        config.RuntimeSetting.w_b);

    auto adjointFunc_Lay2 = adjointFunctionWithMaterial_Lay1(geometry, F, MrInv,
        t_layer_pv_1_vd,
        ac.m_strain_curve,
        ac.m_moduls_curve,
        E, nu, ac.thickness,
        config.RuntimeSetting.w_s,
        config.RuntimeSetting.w_b);


    // Run inverse design optimization
    // Optimizes vertex positions (Vr) and t_layer_2 values to match targetV
    for(int k=0;k<3;k++)
    {
        Vr = sparse_gauss_newton_lay1(geometry, targetV, MrInv,
            t_layer_pv_1_vd, t_layer_pv_2_vd,
            ac.m_strain_curve,
            ac.m_moduls_curve,
            adjointFunc_Lay1, fixedIdx,  
            config.RuntimeSetting.MaxIter,
            config.RuntimeSetting.epsilon,
            config.RuntimeSetting.wM,
            config.RuntimeSetting.wL,
            E, nu, ac.thickness,
            config.RuntimeSetting.w_s,
            config.RuntimeSetting.w_b);

        
        Vr = sparse_gauss_newton_lay2(geometry, targetV, MrInv,
            t_layer_pv_1_vd, t_layer_pv_2_vd,
            ac.m_strain_curve,
            ac.m_moduls_curve,
            adjointFunc_Lay2, fixedIdx,  
            config.RuntimeSetting.MaxIter,
            config.RuntimeSetting.epsilon,
            config.RuntimeSetting.wM,
            config.RuntimeSetting.wL,
            E, nu, ac.thickness,
            config.RuntimeSetting.w_s,
            config.RuntimeSetting.w_b);
    }

    V_pred = Vr;
    V_pred *= 1.0 / scaleFactor;
    std::string output_mesh_inv_path = config.OutputSetting.OutputPath +
        config.ModelSetting.ModelName + "_inv" + ".obj";
    igl::writeOBJ(output_mesh_inv_path, V_pred, F);

    ///***************************************** Output *****************************************///
    spdlog::info("Step 5: Output - Final results from inverse design.");

    // Extract optimized t_layer_2 from VertexData back to VectorXd
    for (auto v : mesh.vertices())
        morph_mesh.t_layer_pv_2[v.getIndex()] = t_layer_pv_2_vd[v];

    // Recompute lambda / kappa from the optimized t layers
    Morphmesh::ComputeMorphingFormTLayers(morph_mesh.t_layer_pv_1, morph_mesh.t_layer_pv_2,
        ac.m_strain_curve, ac.m_moduls_curve, ac.thickness,
        morph_mesh.lambda_pv_s, morph_mesh.kappa_pv_s);

    // Write .obj


    // Write .vtk
    std::vector<Eigen::VectorXd> per_vertex_attributes_list_inv{
        morph_mesh.lambda_pv_t, morph_mesh.kappa_pv_t,
        morph_mesh.lambda_pv_s, morph_mesh.kappa_pv_s,
        morph_mesh.t_layer_pv_1, morph_mesh.t_layer_pv_2
    };
    std::vector<std::string> per_vertex_attributes_str_list_inv{
        "lambda_pv_t", "kappa_pv_t",
        "lambda_pv_s", "kappa_pv_s",
        "t_layer_pv_1", "t_layer_pv_2"
    };
    std::string output_vtk_inv_path = config.OutputSetting.OutputPath +
        config.ModelSetting.ModelName + "_inv" + ".vtk";
    write_output_vtk(output_vtk_inv_path, V_pred, F,
        per_vertex_attributes_list_inv, per_vertex_attributes_str_list_inv);



    ///***************************************** View by Imgui *****************************************///               
    //igl::opengl::glfw::Viewer viewer;
    //viewer.data().set_mesh(V_pred, F);
    ////viewer.data().set_colors(C);
    //viewer.data().show_lines = true;
    //viewer.launch();

    spdlog::info("program finish.");

}

