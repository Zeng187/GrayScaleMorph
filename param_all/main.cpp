// ParamAll: parameterize all patches of a segmented model, compute a shared
// globalScale, and write scaled V / P (and the scale) to disk so Inverse
// and Forward can later read them without recomputing.  Also emits the
// per-patch boundary condition (center-face vertex triple) for the inverse
// design fixed-DOF set.
//
// Outputs under Resources/2_target/{model}/:
//   patch_{i}_V.obj          — V * globalScale, same F as input patch
// Outputs under Resources/2_param/{model}/:
//   patch_{i}_P.obj          — P * globalScale embedded as 3D (z=0), same F
//   global_scale.txt         — single line "<globalScale>"
// Outputs under Resources/2_cond/{model}/:
//   patch_{i}_bound_center.txt — one line "v0 v1 v2" (vertex idx)

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
#include "simulation_utils.h"

int main(int /*argc*/, char* /*argv*/[])
{
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    Config config("cfg.json");
    ActiveComposite ac(config.materialJsonPath());
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    spdlog::info("Param: start.");

    const std::string model       = config.ModelSetting.ModelName;
    const std::string patches_dir = config.PathSetting.SegmentDir + model + "/patches/";
    const std::string param_dir   = config.PathSetting.ParamDir   + model + "/";
    const std::string target_dir  = config.PathSetting.TargetDir  + model + "/";
    const std::string cond_dir    = config.PathSetting.CondDir    + model + "/";
    std::filesystem::create_directories(param_dir);
    std::filesystem::create_directories(target_dir);
    std::filesystem::create_directories(cond_dir);
    spdlog::info("Patches dir : {}", patches_dir);
    spdlog::info("Target out  : {}", target_dir);
    spdlog::info("Param out   : {}", param_dir);
    spdlog::info("Cond out    : {}", cond_dir);

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
        const double scale_i = config.DeviceSetting.platewidth / P_extent;
        spdlog::info("Patch {}: P_extent = {:.4f}, scale_i = {:.6f}", pd.idx, P_extent, scale_i);
        if (scale_i < globalScale) globalScale = scale_i;
    }
    spdlog::info("globalScale = {:.6f}", globalScale);

    // ===== Phase 3: physical scaling + per-patch gauge shift + write to disk =====
    for (const auto& pd : patches) {
        Eigen::MatrixXd V_scaled = pd.V * globalScale;
        Eigen::MatrixXd P_scaled = pd.P * globalScale;

        // Per-patch gauge shift on the scaled geometry.  Only P is shifted.
        {
            const auto pre = computeLambdaStats(V_scaled, pd.F, P_scaled);
            spdlog::info("Patch {} lambda pre-shift : [{:.4f}, {:.4f}], mean {:.4f}",
                         pd.idx, pre.lmin, pre.lmax, pre.lmean);
            const double t = computeGaugeShiftScale(V_scaled, pd.F, P_scaled,
                                                    ac.range_lam.x, ac.range_lam.y);
            if (t > 0.0 && std::isfinite(t) && std::abs(t - 1.0) > 1e-12) {
                P_scaled /= t;
                spdlog::info("Patch {} gauge shift t={:.6f} (P /= t)", pd.idx, t);
            } else {
                spdlog::info("Patch {} gauge shift t={:.6f}, no scaling applied", pd.idx, t);
            }
            const auto post = computeLambdaStats(V_scaled, pd.F, P_scaled);
            spdlog::info("Patch {} lambda post-shift: [{:.4f}, {:.4f}], mean {:.4f}",
                         pd.idx, post.lmin, post.lmax, post.lmean);
            spdlog::info("Patch {} material window  : [{:.4f}, {:.4f}], size {:.4f}",
                         pd.idx, ac.range_lam.x, ac.range_lam.y, ac.range_lam.y - ac.range_lam.x);
        }

        // Embed P (2D) as 3D with z=0 so we can use the OBJ writer
        Eigen::MatrixXd P_obj(P_scaled.rows(), 3);
        P_obj.leftCols(2) = P_scaled;
        P_obj.col(2).setZero();

        const std::string v_path = target_dir + "patch_" + std::to_string(pd.idx) + "_V.obj";
        const std::string p_path = param_dir  + "patch_" + std::to_string(pd.idx) + "_P.obj";
        igl::writeOBJ(v_path, V_scaled, pd.F);
        igl::writeOBJ(p_path, P_obj, pd.F);
        spdlog::info("Patch {} wrote V -> {}", pd.idx, v_path);
        spdlog::info("Patch {} wrote P -> {}", pd.idx, p_path);

        // Boundary condition: 3 vertex indices of the center face (on shifted P)
        std::vector<int> centerV = findCenterVertexIndices(P_scaled, pd.F);
        const std::string cond_path = cond_dir + "patch_" + std::to_string(pd.idx) + "_bound_center.txt";
        std::ofstream ofs(cond_path);
        ofs << centerV[0] << " " << centerV[1] << " " << centerV[2] << "\n";
        spdlog::info("Patch {} wrote cond -> {}", pd.idx, cond_path);
    }

    {
        std::ofstream ofs(param_dir + "global_scale.txt");
        ofs << std::setprecision(17) << globalScale << "\n";
    }
    spdlog::info("Wrote global_scale.txt = {}", globalScale);

    spdlog::info("Param: done.");
    return 0;
}
