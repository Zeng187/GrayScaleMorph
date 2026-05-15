// Param: parameterize a SINGLE mesh and write the result to disk.
//
// Input  (from PathSetting.MeshesDir + ModelName + Postfix):
//   Resources/0_meshes/{model}.obj
//
// Output (to   PathSetting.ParamDir + ModelName/):
//   Resources/3_param/{model}/{model}_V.obj    — scaled V (V * scale)
//   Resources/3_param/{model}/{model}_P.obj    — scaled P (z=0), same F
//   Resources/3_param/{model}/global_scale.txt — single line "<scale>"
//
// "scale" is computed as Platewidth / P_extent so the parameterised mesh
// fills the configured plate width.  For a single mesh there is no
// inter-patch coordination; the multi-patch version (ParamAll) shares a
// global scale across all patches of a segmented model.

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/loop.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <filesystem>

#include <spdlog/spdlog.h>

#include "config.hpp"
#include "material.hpp"
#include "parameterization.h"

int main(int /*argc*/, char* /*argv*/[])
{
    Config config("cfg.json");
    ActiveComposite ac(config.materialJsonPath());
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    spdlog::info("Param (single mesh): start.");

    const std::string model = config.ModelSetting.ModelName;
    const std::string in_path = config.PathSetting.MeshesDir + model + config.ModelSetting.Postfix;
    const std::string out_dir = config.PathSetting.ParamDir   + model + "/";
    std::filesystem::create_directories(out_dir);
    spdlog::info("Input  : {}", in_path);
    spdlog::info("Output : {}", out_dir);

    // Read mesh
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::readOBJ(in_path, V, F)) {
        spdlog::error("Cannot read mesh: {}", in_path);
        return -1;
    }
    size_t nV = V.rows();
    size_t nF = F.rows();
    spdlog::info("Read {} V, {} F.", nV, nF);

    // Loop subdivision if too coarse
    while (nF < config.RuntimeSetting.nFmin) {
        Eigen::MatrixXd tV = V;
        Eigen::MatrixXi tF = F;
        igl::loop(tV, tF, V, F);
        nV = V.rows();
        nF = F.rows();
    }

    // Parameterise (gauge shift applied internally)
    spdlog::info("Parameterising...");
    Eigen::MatrixXd P = parameterization(V, F, ac.range_lam.x, ac.range_lam.y, 0);

    // Single-mesh scale: simply fit Platewidth to the parameterised extent
    const double P_extent = (P.colwise().maxCoeff() - P.colwise().minCoeff()).maxCoeff();
    const double scale    = config.RuntimeSetting.Platewidth / P_extent;
    spdlog::info("P_extent = {:.4f}, scale = {:.6f}", P_extent, scale);

    Eigen::MatrixXd V_scaled = V * scale;
    Eigen::MatrixXd P_scaled = P * scale;

    // Embed P (2D) as 3D with z=0 for OBJ I/O
    Eigen::MatrixXd P_obj(P_scaled.rows(), 3);
    P_obj.leftCols(2) = P_scaled;
    P_obj.col(2).setZero();

    const std::string v_path  = out_dir + model + "_V.obj";
    const std::string p_path  = out_dir + model + "_P.obj";
    const std::string sc_path = out_dir + "global_scale.txt";
    igl::writeOBJ(v_path, V_scaled, F);
    igl::writeOBJ(p_path, P_obj,    F);
    {
        std::ofstream ofs(sc_path);
        ofs << std::setprecision(17) << scale << "\n";
    }
    spdlog::info("Wrote {} / {} / {}", v_path, p_path, sc_path);
    spdlog::info("Param (single mesh): done.");
    return 0;
}
