/// Standalone Parameterize entry point.
///
/// Reads a config JSON, loads the mesh, optionally loop-subdivides, then
/// runs the parameterization pipeline (per-patch or whole-mesh) and writes
/// the resulting 2D param OBJ(s) plus the target shape OBJ.
///
/// Usage:
///   ./Parameterize [--cfg <path>]     (default: cfg.json)

#include <Eigen/Core>

#include <igl/loop.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "config.hpp"
#include "material.hpp"
#include "parameterize_pipeline.h"
#include "patch_utils.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// Expand an Nx2 parameterization matrix to Nx3 (z = 0) for OBJ output.
void writeParamOBJ(const std::string& path,
                   const Eigen::MatrixXd& P,
                   const Eigen::MatrixXi& F)
{
    Eigen::MatrixXd P3d = Eigen::MatrixXd::Zero(P.rows(), 3);
    P3d.col(0) = P.col(0);
    P3d.col(1) = P.col(1);
    igl::writeOBJ(path, P3d, F);
}

/// Per-patch parameterization mode.
///
/// Scales the global mesh to platewidth, extracts each patch by seg_id,
/// parameterizes independently, and writes patch_{pid}_param.obj for every
/// non-empty patch.  Also writes {model}_targ.obj at original scale.
int runPerPatchParam(const Config&          config,
                     ActiveComposite&       ac,
                     const Eigen::MatrixXd& V_global,
                     const Eigen::MatrixXi& F_global,
                     const std::string&     segidPath,
                     const std::string&     paramDir)
{
    const std::vector<int> seg_id = loadSegId(segidPath);
    if (seg_id.empty()) {
        spdlog::error("Segmentation file is empty: {}", segidPath);
        return EXIT_FAILURE;
    }
    if (static_cast<int>(seg_id.size()) != static_cast<int>(F_global.rows())) {
        spdlog::error("seg_id size ({}) != mesh face count ({})",
                      seg_id.size(), F_global.rows());
        return EXIT_FAILURE;
    }

    const double platewidth = config.solver.platewidth;

    // Scale to platewidth (same convention as morph/main.cpp runPerPatchMode)
    const double extent =
        (V_global.colwise().maxCoeff() - V_global.colwise().minCoeff()).maxCoeff();
    const double scaleFactor = platewidth / extent;
    const Eigen::MatrixXd V_scaled = V_global * scaleFactor;

    // Write target at original (unscaled) coordinates
    igl::writeOBJ(paramDir + config.model.name + "_targ.obj", V_global, F_global);
    spdlog::info("Target mesh written to: {}",
                 paramDir + config.model.name + "_targ.obj");

    const int numPatches = *std::max_element(seg_id.begin(), seg_id.end()) + 1;
    spdlog::info("Per-patch mode: {} patches detected.", numPatches);

    for (int pid = 0; pid < numPatches; ++pid) {
        PatchData patch = extractPatch(V_scaled, F_global, seg_id, pid);
        if (patch.F.rows() == 0) {
            spdlog::warn("Patch {} has 0 faces, skipping.", pid);
            continue;
        }

        ParameterizeResult result = parameterizeMesh(
            patch.V, patch.F,
            ac.range_lam.x, ac.range_lam.y,
            platewidth);

        const std::string paramPath =
            paramDir + "patch_" + std::to_string(pid) + "_param.obj";
        writeParamOBJ(paramPath, result.P, result.F);
        spdlog::info("Patch {} parameterized: {} faces -> {}",
                     pid, result.F.rows(), paramPath);
    }

    return EXIT_SUCCESS;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    // ── Parse CLI ──────────────────────────────────────────────────────
    std::string cfgPath = "cfg.json";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--cfg" && i + 1 < argc) {
            cfgPath = argv[++i];
        } else {
            spdlog::error("Usage: {} [--cfg <path>]", argv[0]);
            return EXIT_FAILURE;
        }
    }

    // ── Load configuration & material ──────────────────────────────────
    Config config(cfgPath);
    spdlog::info("Config loaded from: {}", cfgPath);

    ActiveComposite ac(config.material.curves_path);
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();
    spdlog::info("Material curves loaded. lambda range: [{}, {}]",
                 ac.range_lam.x, ac.range_lam.y);

    // ── Prepare output directory ───────────────────────────────────────
    const std::string paramDir = config.paramDir();
    std::filesystem::create_directories(paramDir);

    // ── Load mesh ──────────────────────────────────────────────────────
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::readOBJ(config.model.mesh_path, V, F)) {
        spdlog::error("Could not read mesh: {}", config.model.mesh_path);
        return EXIT_FAILURE;
    }
    spdlog::info("Loaded mesh: {} vertices, {} faces.", V.rows(), F.rows());

    // ── Loop subdivision if face count is below threshold ──────────────
    while (static_cast<int>(F.rows()) < config.solver.nf_min) {
        Eigen::MatrixXd tempV = V;
        Eigen::MatrixXi tempF = F;
        igl::loop(tempV, tempF, V, F);
        spdlog::info("Loop subdivided -> {} vertices, {} faces.", V.rows(), F.rows());
    }

    // ── Check for segmentation file ────────────────────────────────────
    const std::string segidPath =
        config.segmentDir(config.model.name) + "seg_id.txt";

    if (std::filesystem::exists(segidPath)) {
        spdlog::info("Segmentation found: {}", segidPath);
        int ret = runPerPatchParam(config, ac, V, F, segidPath, paramDir);
        spdlog::info("Parameterization complete.");
        return ret;
    }

    spdlog::info("No segmentation found at: {}. Whole-mesh mode.", segidPath);

    // ── Whole-mesh parameterization ────────────────────────────────────

    // Pre-scale to platewidth (matches morph/main.cpp convention)
    const double scaleFactor1 =
        config.solver.platewidth
        / (V.colwise().maxCoeff() - V.colwise().minCoeff()).maxCoeff();
    V *= scaleFactor1;

    ParameterizeResult result = parameterizeMesh(
        V, F,
        ac.range_lam.x, ac.range_lam.y,
        config.solver.platewidth);

    const double totalScale = scaleFactor1 * result.scaleFactor;

    // Write 2D parameterized mesh
    const std::string paramPath = paramDir + config.model.name + "_param.obj";
    writeParamOBJ(paramPath, result.P, result.F);
    spdlog::info("Parameterized mesh written to: {}", paramPath);

    // Write target at original (unscaled) coordinates
    Eigen::MatrixXd V_targ = result.V * (1.0 / totalScale);
    const std::string targPath = paramDir + config.model.name + "_targ.obj";
    igl::writeOBJ(targPath, V_targ, result.F);
    spdlog::info("Target mesh written to: {}", targPath);

    spdlog::info("Parameterization complete.");
    return EXIT_SUCCESS;
}
