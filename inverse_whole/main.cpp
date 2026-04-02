/// InverseWhole — per-patch inverse design for ALL patches in a segmented mesh.
///
/// Reads inverse_whole_cfg.json, requires seg_id.txt.
/// For single-patch inverse design, use the Inverse executable with inverse_cfg.json.

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/loop.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>

#include <spdlog/spdlog.h>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include "config.hpp"
#include "material.hpp"
#include "parameterize_pipeline.h"
#include "simulation_utils.h"
#include "functions.h"
#include "newton.h"
#include "morphmesh.hpp"
#include "morph_functions.hpp"
#include "output.hpp"
#include "patch_utils.h"
#include "inverse_design.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ---------------------------------------------------------------------------
// Run inverse design on a single patch and scatter results to global arrays.
// ---------------------------------------------------------------------------
static void runOnPatch(
    const Eigen::MatrixXd& V_patch,
    const Eigen::MatrixXi& F_patch,
    const Eigen::MatrixXd& P_patch,
    const std::vector<int>& global_face_ids,
    const Config& config,
    const ActiveComposite& ac,
    int patch_id,
    Eigen::VectorXd& global_t1,
    Eigen::VectorXd& global_t2,
    Eigen::VectorXd& global_lam_excess,
    Eigen::VectorXd& global_kap_excess,
    const std::string& morphDir)
{
    const auto& solver = config.solver;
    const int nF_p = static_cast<int>(F_patch.rows());
    spdlog::info("=== Patch {} inverse design: {} vertices, {} faces ===",
                 patch_id, V_patch.rows(), nF_p);

    ManifoldSurfaceMesh mesh_p(F_patch);
    VertexPositionGeometry geom_p(mesh_p, V_patch);
    geom_p.refreshQuantities();

    FaceData<Eigen::Matrix2d> MrInv_p = precomputeMrInv(mesh_p, P_patch, F_patch);
    std::vector<int> fixedIdx_p = findCenterFaceIndices(P_patch, F_patch);

    InverseDesignProblem problem;
    problem.V         = V_patch;
    problem.F         = F_patch;
    problem.P         = P_patch;
    problem.mesh      = &mesh_p;
    problem.geometry  = &geom_p;
    problem.MrInv     = MrInv_p;
    problem.fixedIdx  = fixedIdx_p;
    problem.ac        = &ac;
    problem.max_iter          = solver.max_iter;
    problem.epsilon           = solver.epsilon;
    problem.w_s               = solver.w_s;
    problem.w_b               = solver.w_b;
    problem.wM_kap            = solver.wM_kap;
    problem.wL_kap            = solver.wL_kap;
    problem.wM_lam            = solver.wM_lam;
    problem.wL_lam            = solver.wL_lam;
    problem.wP_kap            = solver.wP_kap;
    problem.wP_lam            = solver.wP_lam;
    problem.penalty_threshold = solver.penalty_threshold;
    problem.betaP             = solver.betaP;
    problem.patch_id          = patch_id;

    InverseDesignResult result = runInverseDesign(problem);

    // Write per-patch projected shape
    {
        std::string path = morphDir + "patch_" + std::to_string(patch_id) + "_proj.obj";
        igl::writeOBJ(path, result.V_proj, F_patch);
        spdlog::info("Patch {}: proj shape -> {}", patch_id, path);
    }

    // Scatter to global arrays
    for (int i = 0; i < nF_p; ++i) {
        int gfid = global_face_ids[i];
        global_t1[gfid]         = result.t1[i];
        global_t2[gfid]         = result.t2[i];
        global_lam_excess[gfid] = result.lam_excess[i];
        global_kap_excess[gfid] = result.kap_excess[i];
    }

    spdlog::info("=== Patch {} done ===", patch_id);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    // --- Config ---
    const std::string cfgPath = "inverse_whole_cfg.json";
    if (!std::filesystem::exists(cfgPath)) {
        spdlog::error("Config not found: {}. InverseWhole requires inverse_whole_cfg.json.", cfgPath);
        return -1;
    }
    Config config(cfgPath);
    const auto& model  = config.model;
    const auto& paths  = config.paths;
    const auto& solver = config.solver;

    // --- Material ---
    ActiveComposite ac(config.material.curves_path);
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    // --- Directories ---
    std::string morphDir   = config.morphDir();
    std::string designDir  = config.designDir();
    std::string metricsDir = paths.output_path + model.name + "/";
    std::filesystem::create_directories(paths.output_path);
    std::filesystem::create_directories(morphDir);
    std::filesystem::create_directories(designDir);
    std::filesystem::create_directories(metricsDir);

    // --- Load mesh ---
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::readOBJ(model.mesh_path, V, F)) {
        spdlog::error("Cannot read mesh: {}", model.mesh_path);
        return -1;
    }
    spdlog::info("Mesh: {} vertices, {} faces.", V.rows(), F.rows());

    size_t nF = F.rows();
    while (nF < static_cast<size_t>(solver.nf_min)) {
        Eigen::MatrixXd tV = V; Eigen::MatrixXi tF = F;
        igl::loop(tV, tF, V, F);
        nF = F.rows();
    }

    // --- Segmentation (required) ---
    std::string segDir = config.segmentDir(model.name);
    std::string segid_path = segDir + "seg_id.txt";

    // Fallback to plain model name
    if (!std::filesystem::exists(segid_path) && !config.segment.method.empty()) {
        segid_path = config.segment.path + model.name + "/seg_id.txt";
    }

    if (!std::filesystem::exists(segid_path)) {
        spdlog::error("seg_id.txt not found at: {}. InverseWhole requires segmentation.", segid_path);
        return -1;
    }

    std::vector<int> seg_id = loadSegId(segid_path);
    int nF_global = static_cast<int>(F.rows());
    if (static_cast<int>(seg_id.size()) != nF_global) {
        spdlog::error("seg_id size ({}) != face count ({})", seg_id.size(), nF_global);
        return -1;
    }

    int num_patches = *std::max_element(seg_id.begin(), seg_id.end()) + 1;
    spdlog::info("Segmentation loaded: {} patches.", num_patches);

    double platewidth = solver.platewidth;

    // --- Scale mesh ---
    Eigen::MatrixXd V_scaled = V;
    double scaleFactor = platewidth / (V_scaled.colwise().maxCoeff() - V_scaled.colwise().minCoeff()).maxCoeff();
    V_scaled *= scaleFactor;

    // --- Global output arrays ---
    Eigen::VectorXd global_t1          = Eigen::VectorXd::Zero(nF_global);
    Eigen::VectorXd global_t2          = Eigen::VectorXd::Zero(nF_global);
    Eigen::VectorXd global_lam_excess  = Eigen::VectorXd::Zero(nF_global);
    Eigen::VectorXd global_kap_excess  = Eigen::VectorXd::Zero(nF_global);

    // --- Write target mesh ---
    {
        Eigen::MatrixXd V_targ = V_scaled * (1.0 / scaleFactor);
        igl::writeOBJ(paths.output_path + model.name + "_targ.obj", V_targ, F);
        igl::writeOBJ(morphDir + model.name + "_targ.obj", V_targ, F);
    }

    // --- Process ALL patches ---
    for (int pid = 0; pid < num_patches; ++pid) {
        PatchData patch = extractPatch(V_scaled, F, seg_id, pid);
        if (patch.F.rows() == 0) {
            spdlog::warn("Patch {} has 0 faces, skipping.", pid);
            continue;
        }
        spdlog::info("Patch {}: {} faces, {} vertices", pid, patch.F.rows(), patch.V.rows());

        ParameterizeResult param = parameterizeMesh(
            patch.V, patch.F, ac.range_lam.x, ac.range_lam.y, platewidth);

        // Write per-patch param mesh
        {
            Eigen::MatrixXd P_3d = Eigen::MatrixXd::Zero(param.P.rows(), 3);
            P_3d.col(0) = param.P.col(0);
            P_3d.col(1) = param.P.col(1);
            std::string path = morphDir + "patch_" + std::to_string(pid) + "_param.obj";
            igl::writeOBJ(path, P_3d, param.F);
        }

        runOnPatch(
            param.V, param.F, param.P,
            patch.global_face_ids,
            config, ac, pid,
            global_t1, global_t2,
            global_lam_excess, global_kap_excess,
            morphDir);
    }

    // --- Merged material output ---
    {
        auto writeMaterial = [&](const std::string& path) {
            std::ofstream ofs(path);
            ofs << "# face_id  t1  t2\n";
            for (int fid = 0; fid < nF_global; ++fid)
                ofs << fid << "  " << global_t1[fid] << "  " << global_t2[fid] << "\n";
        };
        writeMaterial(paths.output_path + model.name + "_material.txt");
        writeMaterial(designDir + model.name + "_material.txt");
        spdlog::info("Merged material written.");
    }

    // --- Merged metrics output ---
    {
        std::string path = metricsDir + "metrics.txt";
        std::ofstream ofs(path);
        ofs << "# face_id  lambda_excess  kappa_excess\n";
        for (int fid = 0; fid < nF_global; ++fid)
            ofs << fid << "  " << global_lam_excess[fid] << "  " << global_kap_excess[fid] << "\n";
        spdlog::info("Merged metrics -> {}", path);
    }

    spdlog::info("InverseWhole finished.");
    return 0;
}
