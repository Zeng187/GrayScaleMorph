// Param: parameterize a SINGLE mesh and write the result + boundary cond.
// Single-mesh case is treated as a one-patch model (patch_0).
//
// Input  (from PathSetting.MeshesDir + ModelName + Postfix):
//   Resources/0_meshes/{model}.obj
//
// Output:
//   Resources/2_target/{model}/patch_0_V.obj         — scaled V (V * scale)
//   Resources/2_param/{model}/patch_0_P.obj          — scaled P (z=0), same F
//   Resources/2_param/{model}/global_scale.txt       — single line "<scale>"
//   Resources/2_cond/{model}/patch_0_bound_center.txt — one line "v0 v1 v2"
//                                                       (vertex idx of center face)
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
#include "simulation_utils.h"

int main(int /*argc*/, char* /*argv*/[])
{
    Config config("cfg.json");
    ActiveComposite ac(config.materialJsonPath());
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    spdlog::info("Param (single mesh): start.");

    const std::string model      = config.ModelSetting.ModelName;
    const std::string in_path    = config.PathSetting.MeshesDir + model + config.ModelSetting.Postfix;
    const std::string param_dir  = config.PathSetting.ParamDir  + model + "/";
    const std::string target_dir = config.PathSetting.TargetDir + model + "/";
    const std::string cond_dir   = config.PathSetting.CondDir   + model + "/";
    std::filesystem::create_directories(param_dir);
    std::filesystem::create_directories(target_dir);
    std::filesystem::create_directories(cond_dir);
    spdlog::info("Input     : {}", in_path);
    spdlog::info("Target out: {}", target_dir);
    spdlog::info("Param out : {}", param_dir);
    spdlog::info("Cond out  : {}", cond_dir);

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

    // Parameterise (pure ARAP, no gauge shift here)
    spdlog::info("Parameterising...");
    Eigen::MatrixXd P = parameterization(V, F, ac.range_lam.x, ac.range_lam.y, 0);

    // Single-mesh scale: fit Platewidth to the parameterised extent
    const double P_extent = (P.colwise().maxCoeff() - P.colwise().minCoeff()).maxCoeff();
    const double scale    = config.DeviceSetting.platewidth / P_extent;
    spdlog::info("P_extent = {:.4f}, scale = {:.6f}", P_extent, scale);

    // Physical scaling first: V_scaled is the device-sized target.
    Eigen::MatrixXd V_scaled = V * scale;
    Eigen::MatrixXd P_scaled = P * scale;

    // Gauge shift on the scaled geometry: move lambda distribution into
    // the material window [lambda_min, lambda_max].  Only P is shifted.
    {
        const auto pre = computeLambdaStats(V_scaled, F, P_scaled);
        spdlog::info("Lambda pre-shift : [{:.4f}, {:.4f}], mean {:.4f}",
                     pre.lmin, pre.lmax, pre.lmean);
        const double t = computeGaugeShiftScale(V_scaled, F, P_scaled,
                                                ac.range_lam.x, ac.range_lam.y);
        if (t > 0.0 && std::isfinite(t) && std::abs(t - 1.0) > 1e-12) {
            P_scaled /= t;
            spdlog::info("Gauge shift t={:.6f}, P /= t (i.e. P *= {:.6f})", t, 1.0 / t);
        } else {
            spdlog::info("Gauge shift t={:.6f}, no scaling applied", t);
        }
        const auto post = computeLambdaStats(V_scaled, F, P_scaled);
        spdlog::info("Lambda post-shift: [{:.4f}, {:.4f}], mean {:.4f}",
                     post.lmin, post.lmax, post.lmean);
        spdlog::info("Material window  : [{:.4f}, {:.4f}]",
                     ac.range_lam.x, ac.range_lam.y);
    }

    // Embed P (2D) as 3D with z=0 for OBJ I/O
    Eigen::MatrixXd P_obj(P_scaled.rows(), 3);
    P_obj.leftCols(2) = P_scaled;
    P_obj.col(2).setZero();

    const std::string v_path  = target_dir + "patch_0_V.obj";
    const std::string p_path  = param_dir  + "patch_0_P.obj";
    const std::string sc_path = param_dir  + "global_scale.txt";
    igl::writeOBJ(v_path, V_scaled, F);
    igl::writeOBJ(p_path, P_obj,    F);
    {
        std::ofstream ofs(sc_path);
        ofs << std::setprecision(17) << scale << "\n";
    }
    spdlog::info("Wrote {} / {} / {}", v_path, p_path, sc_path);

    // ---- Boundary condition: 3 vertex indices of the center face (on P) ----
    {
        std::vector<int> centerV = findCenterVertexIndices(P_scaled, F);
        const std::string cond_path = cond_dir + "patch_0_bound_center.txt";
        std::ofstream ofs(cond_path);
        ofs << centerV[0] << " " << centerV[1] << " " << centerV[2] << "\n";
        spdlog::info("Cond -> {}", cond_path);
    }

    spdlog::info("Param (single mesh): done.");
    return 0;
}
