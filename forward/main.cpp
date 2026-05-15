// Forward: given a parameterization (V, P) and per-face material
// assignment (lambda, kappa or (t1, t2)), run the elastic forward
// simulation and write the deformed mesh.
//
// Inputs under Resources/3_param/{model}/:
//   patch_{i}_V.obj          — scaled target V
//   patch_{i}_P.obj          — scaled 2D parameterization (z=0)
//   global_scale.txt         — globalScale
// Material assignment (TODO: file format to be defined; will read from
// Resources/3_design/{model}/patch_{i}_material.txt).
//
// Outputs under Resources/forward/{model}/:
//   patch_{i}_deformed.obj   — deformed mesh after forward sim
//
// Status: SKELETON ONLY.  Loads cfg, lists target patches, prints TODO.

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

#include <spdlog/spdlog.h>

#include "config.hpp"
#include "material.hpp"

int main(int /*argc*/, char* /*argv*/[])
{
    Config config("cfg.json");
    ActiveComposite ac(config.materialJsonPath());
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    const std::string model = config.ModelSetting.ModelName;
    const std::string param_dir  = config.PathSetting.ParamDir   + model + "/";
    const std::string design_dir = config.PathSetting.DesignDir  + model + "/";
    const std::string out_dir    = config.PathSetting.ForwardDir + model + "/";
    std::filesystem::create_directories(out_dir);

    spdlog::info("Forward: start.");
    spdlog::info("Reads param from  : {}", param_dir);
    spdlog::info("Reads design from : {}", design_dir);
    spdlog::info("Writes results to : {}", out_dir);

    if (!std::filesystem::exists(param_dir)) {
        spdlog::error("Param output not found.  Run Param first.");
        return -1;
    }

    spdlog::warn("Forward: implementation skeleton only.  "
                 "Per-patch forward simulation not yet wired.");
    spdlog::warn("Once Inverse writes Resources/3_design/{}/patch_*_material.txt, "
                 "this entry should load (V, P, material) per patch, build simFunc, "
                 "run forward newton, and write deformed mesh.", model);

    spdlog::info("Forward: done (no-op).");
    return 0;
}
