/// Inverse — single-patch inverse design.
///
/// Reads inverse_cfg.json (requires patch.id >= 0 and seg_id.txt).
/// For all-patches inverse design, use InverseWhole with inverse_whole_cfg.json.

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/loop.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <set>

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

namespace {
/// Convert DOF indices (3 per vertex) to unique vertex indices.
std::vector<int> fixedDofsToVertices(const std::vector<int>& fixedIdx) {
    std::set<int> vset;
    for (int dof : fixedIdx) if (dof >= 0) vset.insert(dof / 3);
    return {vset.begin(), vset.end()};
}
/// Write boundary condition file (3 vertex indices on one line).
void writeCondFile(const std::string& path, const std::vector<int>& fixedIdx) {
    auto verts = fixedDofsToVertices(fixedIdx);
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream ofs(path);
    for (size_t i = 0; i < verts.size(); i++)
        ofs << verts[i] << (i + 1 < verts.size() ? " " : "\n");
    spdlog::info("Boundary condition written to: {}", path);
}
} // anonymous namespace

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    // --- Config ---
    const std::string cfgPath = "inverse_cfg.json";
    if (!std::filesystem::exists(cfgPath)) {
        spdlog::error("Config not found: {}. Inverse requires inverse_cfg.json.", cfgPath);
        return -1;
    }
    Config config(cfgPath);
    const auto& model  = config.model;
    const auto& solver = config.solver;

    if (config.patch.id < 0) {
        spdlog::error("patch.id must be >= 0 in inverse_cfg.json. "
                      "For all-patches mode, use InverseWhole.");
        return -1;
    }
    const int target_pid = config.patch.id;

    // --- Material ---
    ActiveComposite ac(config.material.curves_path);
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    // --- Directories ---
    std::string morphDir   = config.morphDir();
    std::string designDir  = config.designDir();
    std::filesystem::create_directories(morphDir);
    std::filesystem::create_directories(designDir);

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
    std::string segDir = config.segmentDir();
    std::string segid_path = segDir + "seg_id.txt";
    if (!std::filesystem::exists(segid_path) && !config.segment.method.empty())
        segid_path = config.segment.path + model.name + "/seg_id.txt";

    if (!std::filesystem::exists(segid_path)) {
        spdlog::error("seg_id.txt not found at: {}. Single-patch Inverse requires segmentation.", segid_path);
        return -1;
    }

    std::vector<int> seg_id = loadSegId(segid_path);
    int nF_global = static_cast<int>(F.rows());
    if (static_cast<int>(seg_id.size()) != nF_global) {
        spdlog::error("seg_id size ({}) != face count ({})", seg_id.size(), nF_global);
        return -1;
    }

    int num_patches = *std::max_element(seg_id.begin(), seg_id.end()) + 1;
    if (target_pid >= num_patches) {
        spdlog::error("patch.id={} but only {} patches available.", target_pid, num_patches);
        return -1;
    }
    spdlog::info("Single-patch mode: processing patch {} of {}.", target_pid, num_patches);

    // --- Scale mesh ---
    double platewidth = solver.platewidth;
    Eigen::MatrixXd V_scaled = V;
    double scaleFactor = platewidth / (V_scaled.colwise().maxCoeff() - V_scaled.colwise().minCoeff()).maxCoeff();
    V_scaled *= scaleFactor;

    // --- Extract patch ---
    PatchData patch = extractPatch(V_scaled, F, seg_id, target_pid);
    if (patch.F.rows() == 0) {
        spdlog::error("Patch {} has 0 faces (label may not exist in seg_id).", target_pid);
        return -1;
    }
    spdlog::info("Patch {}: {} faces, {} vertices", target_pid, patch.F.rows(), patch.V.rows());

    // --- Parameterize ---
    ParameterizeResult param = parameterizeMesh(
        patch.V, patch.F, ac.range_lam.x, ac.range_lam.y, platewidth);

    // Write param mesh
    {
        Eigen::MatrixXd P_3d = Eigen::MatrixXd::Zero(param.P.rows(), 3);
        P_3d.col(0) = param.P.col(0);
        P_3d.col(1) = param.P.col(1);
        std::string path = morphDir + "patch_" + std::to_string(target_pid) + "_param.obj";
        igl::writeOBJ(path, P_3d, param.F);
        spdlog::info("Param mesh -> {}", path);
    }

    // --- Inverse design ---
    ManifoldSurfaceMesh mesh_p(param.F);
    VertexPositionGeometry geom_p(mesh_p, param.V);
    geom_p.refreshQuantities();

    FaceData<Eigen::Matrix2d> MrInv_p = precomputeMrInv(mesh_p, param.P, param.F);
    std::vector<int> fixedIdx_p = findCenterFaceIndices(param.P, param.F);
    writeCondFile(config.condDir() + "bound_center.txt", fixedIdx_p);

    InverseDesignProblem problem;
    problem.V         = param.V;
    problem.F         = param.F;
    problem.P         = param.P;
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
    problem.patch_id          = target_pid;

    InverseDesignResult result = runInverseDesign(problem);

    // --- Output ---
    int nF_patch = static_cast<int>(param.F.rows());

    // Projected shape
    {
        std::string path = morphDir + "patch_" + std::to_string(target_pid) + "_proj.obj";
        igl::writeOBJ(path, result.V_proj, param.F);
        spdlog::info("Proj shape -> {}", path);
    }

    // Material (patch-local face ids)
    {
        std::string path = designDir + "patch_" + std::to_string(target_pid) + "_material.txt";
        std::ofstream ofs(path);
        ofs << "# local_face_id  t1  t2  (global_face_id)\n";
        for (int i = 0; i < nF_patch; ++i) {
            ofs << i << "  " << result.t1[i] << "  " << result.t2[i]
                << "  " << patch.global_face_ids[i] << "\n";
        }
        spdlog::info("Material -> {}", path);
    }

    // Metrics
    {
        std::string path = morphDir + "patch_" + std::to_string(target_pid) + "_metrics.txt";
        std::ofstream ofs(path);
        ofs << "# local_face_id  lambda_excess  kappa_excess\n";
        for (int i = 0; i < nF_patch; ++i) {
            ofs << i << "  " << result.lam_excess[i] << "  " << result.kap_excess[i] << "\n";
        }
        spdlog::info("Metrics -> {}", path);
    }

    spdlog::info("Inverse patch {} finished. dist_proj={:.6f}", target_pid, result.dist_proj);
    return 0;
}
