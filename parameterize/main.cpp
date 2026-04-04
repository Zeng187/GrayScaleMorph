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
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "config.hpp"
#include "material.hpp"
#include "parameterization.h"
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

/// Jet colormap: maps t in [0,1] to (R,G,B) in [0,255].
std::array<unsigned char, 3> jetColormap(double t)
{
    t = std::clamp(t, 0.0, 1.0);
    const double r = std::clamp(1.5 - std::abs(4.0 * t - 3.0), 0.0, 1.0);
    const double g = std::clamp(1.5 - std::abs(4.0 * t - 2.0), 0.0, 1.0);
    const double b = std::clamp(1.5 - std::abs(4.0 * t - 1.0), 0.0, 1.0);
    return {
        static_cast<unsigned char>(std::lround(255.0 * r)),
        static_cast<unsigned char>(std::lround(255.0 * g)),
        static_cast<unsigned char>(std::lround(255.0 * b))
    };
}

/// Write per-face distortion metrics and a colored PLY for visualization.
///
/// Outputs:
///   {prefix}_distortion.txt  — per-face: face_id sigma1 sigma2 conformal_ratio area_scale
///   {prefix}_colored.ply     — 3D mesh with vertex colors from conformal_ratio (jet colormap)
/// @param V          3D vertices (for SVD computation; may be at platewidth scale).
/// @param P          2D parameterization (same scale as V).
/// @param F          Face matrix.
/// @param prefix     Output file prefix (path without extension).
/// @param V_display  Optional vertices for PLY coordinates (e.g. original scale).
///                   If null, V is used for PLY output.
/// @param fixedCmapMax  If > 0, use this as colormap upper bound (for cross-run consistency).
///                      If <= 0, use data-driven max.
void writeDistortionOutput(const Eigen::MatrixXd& V,
                           const Eigen::MatrixXd& P,
                           const Eigen::MatrixXi& F,
                           const std::string&     prefix,
                           const Eigen::MatrixXd* V_display = nullptr,
                           double fixedCmapMax = 0.0)
{
    if (V.rows() == 0 || P.rows() == 0 || F.rows() == 0) return;

    const auto [sigma1, sigma2, angles] = computeSVDdata(V, P, F);
    (void)angles;

    const Eigen::MatrixXd& Vout = V_display ? *V_display : V;
    const int nV = static_cast<int>(Vout.rows());
    const int nF = static_cast<int>(F.rows());
    constexpr double kEps = 1e-16;

    // Per-face metrics (use Vout for face areas to match display geometry)
    Eigen::VectorXd faceArea(nF), confRatio(nF), areaScale(nF);
    for (int fi = 0; fi < nF; ++fi) {
        const Eigen::Vector3d v0 = Vout.row(F(fi, 0));
        const Eigen::Vector3d v1 = Vout.row(F(fi, 1));
        const Eigen::Vector3d v2 = Vout.row(F(fi, 2));
        faceArea(fi) = 0.5 * (v1 - v0).cross(v2 - v0).norm();

        const double s1 = std::max(sigma1(fi), sigma2(fi));
        const double s2 = std::max(kEps, std::min(sigma1(fi), sigma2(fi)));
        confRatio(fi)  = s1 / s2;
        areaScale(fi)  = sigma1(fi) * sigma2(fi);
    }

    // Write per-face text file
    {
        std::ofstream out(prefix + "_distortion.txt");
        out << std::setprecision(10)
            << "# face_id sigma1 sigma2 conformal_ratio area_scale\n";
        for (int fi = 0; fi < nF; ++fi) {
            out << fi
                << " " << sigma1(fi)
                << " " << sigma2(fi)
                << " " << confRatio(fi)
                << " " << areaScale(fi) << "\n";
        }
    }

    // Face → vertex: area-weighted average of conformal_ratio
    Eigen::VectorXd vtxValSum  = Eigen::VectorXd::Zero(nV);
    Eigen::VectorXd vtxWgtSum  = Eigen::VectorXd::Zero(nV);
    for (int fi = 0; fi < nF; ++fi) {
        const double w = std::max(faceArea(fi), kEps);
        for (int c = 0; c < 3; ++c) {
            vtxValSum(F(fi, c)) += w * confRatio(fi);
            vtxWgtSum(F(fi, c)) += w;
        }
    }

    // Colormap range: fixed if specified, otherwise data-driven
    const double cmapMin = 1.0;
    const double cmapMax = (fixedCmapMax > cmapMin)
        ? fixedCmapMax
        : std::max(confRatio.maxCoeff(), 1.01);
    const double cmapSpan = cmapMax - cmapMin;
    spdlog::info("Colormap range: [{:.6f}, {:.6f}]", cmapMin, cmapMax);

    // Write colored PLY (ASCII)
    std::ofstream ply(prefix + "_colored.ply");
    ply << "ply\n"
        << "format ascii 1.0\n"
        << "comment conformal_ratio colormap range: "
        << cmapMin << " " << cmapMax << "\n"
        << "element vertex " << nV << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "element face " << nF << "\n"
        << "property list uchar int vertex_indices\n"
        << "end_header\n";

    for (int vi = 0; vi < nV; ++vi) {
        double ratio = 1.0;
        if (vtxWgtSum(vi) > kEps) {
            ratio = vtxValSum(vi) / vtxWgtSum(vi);
        }
        const double t = std::clamp((ratio - cmapMin) / cmapSpan, 0.0, 1.0);
        const auto rgb = jetColormap(t);
        const double z = (Vout.cols() > 2) ? Vout(vi, 2) : 0.0;
        ply << std::fixed << std::setprecision(6)
            << Vout(vi, 0) << " " << Vout(vi, 1) << " " << z << " "
            << static_cast<int>(rgb[0]) << " "
            << static_cast<int>(rgb[1]) << " "
            << static_cast<int>(rgb[2]) << "\n";
    }
    for (int fi = 0; fi < nF; ++fi) {
        ply << "3 " << F(fi, 0) << " " << F(fi, 1) << " " << F(fi, 2) << "\n";
    }
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
                     const std::string&     paramDir,
                     double                 cmapMax = 0.0)
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

        // Extract original-scale patch for PLY display coordinates
        PatchData patchOrig = extractPatch(V_global, F_global, seg_id, pid);
        writeDistortionOutput(result.V, result.P, result.F,
                              paramDir + "patch_" + std::to_string(pid),
                              &patchOrig.V, cmapMax);
        spdlog::info("Patch {} distortion diagnostics written.", pid);
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
    double cmapMax = 0.0;  // 0 = data-driven
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cfg" && i + 1 < argc) {
            cfgPath = argv[++i];
        } else if (arg == "--cmap-max" && i + 1 < argc) {
            cmapMax = std::atof(argv[++i]);
        } else {
            spdlog::error("Usage: {} [--cfg <path>] [--cmap-max <value>]", argv[0]);
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
        config.segmentDir() + "seg_id.txt";

    if (std::filesystem::exists(segidPath)) {
        spdlog::info("Segmentation found: {}", segidPath);
        int ret = runPerPatchParam(config, ac, V, F, segidPath, paramDir, cmapMax);
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

    writeDistortionOutput(result.V, result.P, result.F,
                          paramDir + config.model.name, &V_targ, cmapMax);
    spdlog::info("Whole-mesh distortion diagnostics written.");

    spdlog::info("Parameterization complete.");
    return EXIT_SUCCESS;
}
