/// InverseWhole — per-patch inverse design for ALL patches.
///
/// Reads patch meshes directly from segmentDir()/patches/patch_{pid}.obj.
/// Each patch is independently parameterized, inverse-designed, and output.

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/loop.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>
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
#include "inverse_design.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace {

std::vector<int> fixedDofsToVertices(const std::vector<int>& fixedIdx) {
    std::set<int> vset;
    for (int dof : fixedIdx) if (dof >= 0) vset.insert(dof / 3);
    return {vset.begin(), vset.end()};
}

void writeCondFile(const std::string& path, const std::vector<int>& fixedIdx) {
    auto verts = fixedDofsToVertices(fixedIdx);
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream ofs(path);
    for (size_t i = 0; i < verts.size(); i++)
        ofs << verts[i] << (i + 1 < verts.size() ? " " : "\n");
    spdlog::info("Boundary condition written to: {}", path);
}

/// Discover patch OBJ files in a directory, sorted by patch id.
/// Expects files named patch_0.obj, patch_1.obj, ...
std::vector<std::string> discoverPatches(const std::string& patchesDir) {
    std::vector<std::pair<int, std::string>> found;
    const std::regex pat(R"(patch_(\d+)\.obj)");
    for (auto& entry : std::filesystem::directory_iterator(patchesDir)) {
        std::smatch m;
        std::string fname = entry.path().filename().string();
        if (std::regex_match(fname, m, pat))
            found.emplace_back(std::stoi(m[1].str()), entry.path().string());
    }
    std::sort(found.begin(), found.end());
    std::vector<std::string> paths;
    for (auto& [id, p] : found) paths.push_back(std::move(p));
    return paths;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    // --- Config ---
    const std::string cfgPath = "inverse_whole_cfg.json";
    if (!std::filesystem::exists(cfgPath)) {
        spdlog::error("Config not found: {}", cfgPath);
        return -1;
    }
    Config config(cfgPath);
    const auto& model  = config.model;
    const auto& solver = config.solver;

    // --- Material ---
    ActiveComposite ac(config.material.curves_path);
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    // --- Output directories ---
    std::string morphDir   = config.morphDir();
    std::string designDir  = config.designDir();
    std::string condDir    = config.condDir();
    std::filesystem::create_directories(morphDir);
    std::filesystem::create_directories(designDir);
    std::filesystem::create_directories(condDir);

    // --- Discover patches ---
    std::string patchesDir = config.segmentDir() + "patches/";
    if (!std::filesystem::is_directory(patchesDir)) {
        spdlog::error("Patches directory not found: {}", patchesDir);
        return -1;
    }
    std::vector<std::string> patchFiles = discoverPatches(patchesDir);
    int numPatches = static_cast<int>(patchFiles.size());
    if (numPatches == 0) {
        spdlog::error("No patch_*.obj files found in: {}", patchesDir);
        return -1;
    }
    spdlog::info("Found {} patches in: {}", numPatches, patchesDir);

    // --- Load all patches and find consistent scale ---
    struct PatchMesh { Eigen::MatrixXd V; Eigen::MatrixXi F; };
    std::vector<PatchMesh> patches(numPatches);
    double maxExtent = 0.0;
    int largestPatch = 0;

    for (int pid = 0; pid < numPatches; ++pid) {
        if (!igl::readOBJ(patchFiles[pid], patches[pid].V, patches[pid].F)) {
            spdlog::error("Cannot read patch: {}", patchFiles[pid]);
            return -1;
        }
        // Loop subdivide per-patch if below nf_min
        while (static_cast<int>(patches[pid].F.rows()) < solver.nf_min) {
            Eigen::MatrixXd tV = patches[pid].V;
            Eigen::MatrixXi tF = patches[pid].F;
            igl::loop(tV, tF, patches[pid].V, patches[pid].F);
        }
        double ext = (patches[pid].V.colwise().maxCoeff()
                    - patches[pid].V.colwise().minCoeff()).maxCoeff();
        if (ext > maxExtent) {
            maxExtent = ext;
            largestPatch = pid;
        }
        spdlog::info("Patch {}: {} vertices, {} faces (from {})",
                     pid, patches[pid].V.rows(), patches[pid].F.rows(),
                     patchFiles[pid]);
    }

    double platewidth = solver.platewidth;
    double globalScale = platewidth / maxExtent;
    spdlog::info("Global scale: {:.6f} (largest patch {} extent {:.4f} -> platewidth {})",
                 globalScale, largestPatch, maxExtent, platewidth);

    // --- Process each patch ---
    for (int pid = 0; pid < numPatches; ++pid) {
        Eigen::MatrixXd V_scaled = patches[pid].V * globalScale;
        const Eigen::MatrixXi& F_patch = patches[pid].F;
        const int nF_patch = static_cast<int>(F_patch.rows());

        spdlog::info("=== Patch {} inverse design: {} vertices, {} faces ===",
                     pid, V_scaled.rows(), nF_patch);

        // Parameterize
        ParameterizeResult param = parameterizeMesh(
            V_scaled, F_patch, ac.range_lam.x, ac.range_lam.y, platewidth);

        double invTotalScale = 1.0 / param.scaleFactor;

        // Write param mesh (2D)
        {
            Eigen::MatrixXd P_3d = Eigen::MatrixXd::Zero(param.P.rows(), 3);
            P_3d.col(0) = param.P.col(0);
            P_3d.col(1) = param.P.col(1);
            igl::writeOBJ(morphDir + "patch_" + std::to_string(pid) + "_param.obj",
                          P_3d, param.F);
        }

        // Inverse design
        ManifoldSurfaceMesh mesh_p(param.F);
        VertexPositionGeometry geom_p(mesh_p, param.V);
        geom_p.refreshQuantities();

        FaceData<Eigen::Matrix2d> MrInv_p = precomputeMrInv(mesh_p, param.P, param.F);
        std::vector<int> fixedIdx_p = findCenterFaceIndices(param.P, param.F);
        writeCondFile(condDir + "patch_" + std::to_string(pid) + "_bound_center.txt",
                      fixedIdx_p);

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
        problem.patch_id          = pid;

        InverseDesignResult result = runInverseDesign(problem);

        // Write proj shape (rescaled to original coordinates)
        {
            Eigen::MatrixXd V_proj = result.V_proj * invTotalScale;
            igl::writeOBJ(morphDir + "patch_" + std::to_string(pid) + "_proj.obj",
                          V_proj, param.F);
        }

        // Write per-patch material
        {
            std::string matPath = designDir + "patch_" + std::to_string(pid) + "_material.txt";
            std::ofstream ofs(matPath);
            ofs << "# face_id  t1  t2\n";
            for (int i = 0; i < nF_patch; ++i)
                ofs << i << "  " << result.t1[i] << "  " << result.t2[i] << "\n";
            spdlog::info("Patch {} material -> {}", pid, matPath);
        }

        // Write per-patch metrics
        {
            std::string metPath = morphDir + "patch_" + std::to_string(pid) + "_metrics.txt";
            std::ofstream ofs(metPath);
            ofs << "# face_id  lambda_excess  kappa_excess\n";
            for (int i = 0; i < nF_patch; ++i)
                ofs << i << "  " << result.lam_excess[i] << "  " << result.kap_excess[i] << "\n";
        }

        spdlog::info("=== Patch {} done (dist_proj={:.6f}) ===", pid, result.dist_proj);
    }

    spdlog::info("InverseWhole finished: {} patches processed.", numPatches);
    return 0;
}
