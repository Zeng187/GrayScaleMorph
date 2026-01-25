// IglViewer.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <igl/readOBJ.h>     
#include <igl/writeOBJ.h>
#include <igl/opengl/glfw/Viewer.h>   
#include <igl/file_dialog_open.h>
#include <igl/read_triangle_mesh.h>

#include <iostream>
#include <fstream>  
#include <vector>    
#include <string> 
#include <io.h>

#include <spdlog/spdlog.h>

#include"config.hpp"
#include "material.hpp"

int main(int argc, char* argv[])
{

    Config config("cfg.json");
	Grayscale_Material gray_material(config.ResourceSetting.MaterialPath);

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    spdlog::info("program start:");

    std::string input_mesh_path = config.ModelSetting.InputPath +  config.ModelSetting.ModelName + config.ModelSetting.Postfix;
    if (!igl::readOBJ(input_mesh_path, V, F)) {
        spdlog::error("Error: Could not read output_mesh.obj\n");
        return -1;
    }

    size_t nV = V.rows();
    size_t nF = F.rows();
    spdlog::info("Read meshes with {0} vertices and {1} faces.",nV,nF);

    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    //viewer.data().set_colors(C);
    viewer.data().show_lines = true;
    viewer.launch();

    spdlog::info("program finish.");

}

