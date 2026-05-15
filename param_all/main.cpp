// Param: parameterize all patches of a segmented model, compute a shared
// globalScale, and write scaled V / P (and the scale) to disk so Inverse
// and Forward can later read them without recomputing.
//
// Outputs under Resources/3_param/{model}/:
//   patch_{i}_V.obj          — V * globalScale, same F as input patch
//   patch_{i}_P.obj          — P * globalScale embedded as 3D (z=0), same F
//   global_scale.txt         — single line "<globalScale>"

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/loop.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <filesystem>

#include <spdlog/spdlog.h>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include "config.hpp"
#include "material.hpp"
#include "parameterization.h"

int main(int /*argc*/, char* /*argv*/[])
{
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    Config config("cfg.json");
    ActiveComposite ac(config.materialJsonPath());
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    spdlog::info("Param: start.");

    const std::string model = config.ModelSetting.ModelName;
    const std::string patches_dir = config.PathSetting.SegmentDir + model + "_original_planA/patches/";
    const std::string out_dir     = config.PathSetting.ParamDir   + model + "/";
    std::filesystem::create_directories(out_dir);
    spdlog::info("Patches dir : {}", patches_dir);
    spdlog::info("Output dir  : {}", out_dir);

    // ===== Phase 1: read & parameterize each patch =====
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

        while (nF < config.RuntimeSetting.nFmin) {
            Eigen::MatrixXd tV = Vp;
            Eigen::MatrixXi tF = Fp;
            igl::loop(tV, tF, Vp, Fp);
            nV = Vp.rows();
            nF = Fp.rows();
        }

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

    if (patches.empty()) {
        spdlog::error("No patches found, abort.");
        return -1;
    }

    // ===== Phase 2: globalScale = min(platewidth / P_extent_i) =====
    double globalScale = std::numeric_limits<double>::infinity();
    for (const auto& pd : patches) {
        const double P_extent = (pd.P.colwise().maxCoeff() - pd.P.colwise().minCoeff()).maxCoeff();
        const double scale_i = config.RuntimeSetting.Platewidth / P_extent;
        spdlog::info("Patch {}: P_extent = {:.4f}, scale_i = {:.6f}", pd.idx, P_extent, scale_i);
        if (scale_i < globalScale) globalScale = scale_i;
    }
    spdlog::info("globalScale = {:.6f}", globalScale);

    // ===== Phase 3: write scaled V, P, and globalScale to disk =====
    for (const auto& pd : patches) {
        Eigen::MatrixXd V_scaled = pd.V * globalScale;
        Eigen::MatrixXd P_scaled = pd.P * globalScale;

        // Embed P (2D) as 3D with z=0 so we can use the OBJ writer
        Eigen::MatrixXd P_obj(P_scaled.rows(), 3);
        P_obj.leftCols(2) = P_scaled;
        P_obj.col(2).setZero();

        const std::string v_path = out_dir + "patch_" + std::to_string(pd.idx) + "_V.obj";
        const std::string p_path = out_dir + "patch_" + std::to_string(pd.idx) + "_P.obj";
        igl::writeOBJ(v_path, V_scaled, pd.F);
        igl::writeOBJ(p_path, P_obj, pd.F);
        spdlog::info("Patch {} wrote V → {}", pd.idx, v_path);
        spdlog::info("Patch {} wrote P → {}", pd.idx, p_path);
    }

    {
        std::ofstream ofs(out_dir + "global_scale.txt");
        ofs << std::setprecision(17) << globalScale << "\n";
    }
    spdlog::info("Wrote global_scale.txt = {}", globalScale);

    spdlog::info("Param: done.");
    return 0;
}
