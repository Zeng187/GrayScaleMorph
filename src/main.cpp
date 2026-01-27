// IglViewer.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <igl/readOBJ.h>     
#include <igl/writeOBJ.h>
#include <igl/opengl/glfw/Viewer.h>   
#include <igl/file_dialog_open.h>
#include <igl/read_triangle_mesh.h>
#include <igl/loop.h>
#include <iostream>
#include <fstream>  
#include <vector>    
#include <string> 
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

int main(int argc, char* argv[])
{


    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    Config config("cfg.json");
    ActiveComposite ac(config.ResourceSetting.MaterialPath);

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
    morph_mesh.ComputeMorphophing(geometry, V, F,
        boundary_vertex_flags, boundary_face_flags, boundary_ref_indices,
        MrInv, morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t, morph_mesh.kappa_pv_t,
        morph_mesh.kappa_pf_t);


	// set the inital morphophing parameters with limits
    for (int r = 0; r < nF; r++)
    {
        double lam = morph_mesh.lambda_pf_t[r];
        double kap = morph_mesh.kappa_pf_t[r];
        morph_mesh.lambda_pf_s[r] = std::min(std::max(lam, ac.range_lam.x), ac.range_lam.y);
        morph_mesh.kappa_pf_s[r]  = std::min(std::max(kap, ac.range_kap.x), ac.range_kap.y);
    }
    for (int r = 0; r < nV; r++)
    {
        double lam = morph_mesh.lambda_pv_t[r];
        double kap = morph_mesh.kappa_pv_t[r];
        morph_mesh.lambda_pv_s[r] = std::min(std::max(lam, ac.range_lam.x), ac.range_lam.y);
        morph_mesh.kappa_pv_s[r]  = std::min(std::max(kap, ac.range_kap.x), ac.range_kap.y);
    }




    ///***************************************** Forward Predit *****************************************///

    spdlog::info("Step 3: Forward Predit.");

    FaceData<double> lambda_pf_s(mesh, morph_mesh.lambda_pf_s);
    FaceData<double> kappa_pf_s(mesh, morph_mesh.kappa_pf_s);
    VertexData<double> lambda_pv_s(mesh, morph_mesh.lambda_pv_s);
    VertexData<double> kappa_pv_s(mesh, morph_mesh.kappa_pv_s);
    FaceData<bool> is_boundary_face(mesh);
    FaceData<int> boundary_ref_index(mesh);
    VertexData<bool> is_boundary_vertex(mesh);

    auto func = simulationFunction(geometry,
        MrInv, 
        lambda_pv_s,
        kappa_pv_s, 
		E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b
        );


    auto Vr = V;
    newton(geometry, Vr, func, config.RuntimeSetting.MaxIter, 
        config.RuntimeSetting.epsilon, true, fixedIdx);


    ///***************************************** Output *****************************************///
    spdlog::info("Step 4: Output.");
    
    auto V_pred = Vr;
	V_pred *= 1.0 / scaleFactor;

	std::string output_mesh_pred_path = config.OutputSetting.OutputPath + 
        config.ModelSetting.ModelName + "_pred" + ".obj";
    igl::writeOBJ(output_mesh_pred_path, V_pred, F);


    morph_mesh.ComputeMorphophing(geometry, Vr, F, boundary_vertex_flags, boundary_face_flags, boundary_ref_indices,
        MrInv, morph_mesh.lambda_pv_r, morph_mesh.lambda_pf_r, morph_mesh.kappa_pv_r,
        morph_mesh.kappa_pf_r);

	std::vector <Eigen::VectorXd> per_vertex_attributes_list{
		morph_mesh.lambda_pv_t, morph_mesh.kappa_pv_t,
        morph_mesh.lambda_pv_s, morph_mesh.kappa_pv_s,
		morph_mesh.lambda_pv_r, morph_mesh.kappa_pv_r
    };
    std::vector <std::string> per_vertex_attributes_str_list{
       "lambda_pv_t", "kappa_pv_t",
       "lambda_pv_s", "kappa_pv_s",
       "lambda_pv_r", "kappa_pv_r"
    };
    std::string output_vtk_pred_path = config.OutputSetting.OutputPath + 
        config.ModelSetting.ModelName + "_pred" + ".vtk";
    write_output_vtk(output_vtk_pred_path, V_pred, F,
                      per_vertex_attributes_list, per_vertex_attributes_str_list);


    ///***************************************** View by Imgui *****************************************///               
    //igl::opengl::glfw::Viewer viewer;
    //viewer.data().set_mesh(V_pred, F);
    ////viewer.data().set_colors(C);
    //viewer.data().show_lines = true;
    //viewer.launch();

    spdlog::info("program finish.");

}

