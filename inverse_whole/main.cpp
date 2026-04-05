/// InverseWhole — per-patch inverse design for ALL patches.
///
/// Three-phase pipeline:
///   Phase 1: Read patches → parameterize (gauge shift only, no V scaling)
///   Phase 2: PCA-rotate each P → compute per-patch scale → global scale = min
///   Phase 3: Apply global scale to V and P → inverse design → output

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>
#include <set>
#include <cfloat>

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

/// PCA-rotate 2D points so the longest principal axis aligns with x.
/// Returns the rotated P and the length of the longest axis (x-extent after rotation).
std::pair<Eigen::MatrixXd, double> pcaAlignP(const Eigen::MatrixXd& P)
{
    // Center
    Eigen::RowVector2d center = P.colwise().mean();
    Eigen::MatrixXd Pc = P.rowwise() - center;

    // 2x2 covariance
    Eigen::Matrix2d C = Pc.transpose() * Pc / static_cast<double>(Pc.rows());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(C);

    // Eigenvectors sorted ascending — last column = largest variance direction
    Eigen::Matrix2d Q = eig.eigenvectors();
    // We want the largest eigenvector as x-axis: swap columns if needed
    if (eig.eigenvalues()(0) > eig.eigenvalues()(1))
        Q.col(0).swap(Q.col(1));

    // Ensure right-handed (det > 0)
    if (Q.determinant() < 0.0)
        Q.col(1) *= -1.0;

    // Rotate: each row of Pc gets multiplied by Q
    Eigen::MatrixXd P_rot = Pc * Q;

    // Longest axis = x-extent after rotation
    double xExtent = P_rot.col(0).maxCoeff() - P_rot.col(0).minCoeff();

    // Translate back to centred (keep centered for clean output)
    return {P_rot, xExtent};
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
    std::filesystem::create_directories(config.paramDir());

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

    // =====================================================================
    // Phase 1: Load patches → parameterize (gauge shift only, no V scaling)
    // =====================================================================
    struct PatchParam {
        Eigen::MatrixXd V;   // 3D target (original scale)
        Eigen::MatrixXi F;
        Eigen::MatrixXd P;   // 2D parameterization (gauge-shifted, then PCA-rotated)
    };
    std::vector<PatchParam> patchParams(numPatches);

    for (int pid = 0; pid < numPatches; ++pid) {
        Eigen::MatrixXd V_raw;
        Eigen::MatrixXi F_raw;
        if (!igl::readOBJ(patchFiles[pid], V_raw, F_raw)) {
            spdlog::error("Cannot read patch: {}", patchFiles[pid]);
            return -1;
        }
        while (static_cast<int>(F_raw.rows()) < solver.nf_min)
            linearSubdivide(V_raw, F_raw);

        // Parameterize at original scale (only P gauge-shifted)
        ParameterizeResult param = parameterizeMesh(
            V_raw, F_raw, ac.range_lam.x, ac.range_lam.y);

        patchParams[pid].V = std::move(param.V);
        patchParams[pid].F = std::move(param.F);
        patchParams[pid].P = std::move(param.P);

        spdlog::info("Patch {}: {} vertices, {} faces (from {})",
                     pid, patchParams[pid].V.rows(), patchParams[pid].F.rows(),
                     patchFiles[pid]);
    }

    // =====================================================================
    // Phase 2: PCA-rotate each P, compute globalScale from P extents
    // =====================================================================
    double globalScale = DBL_MAX;
    int constrainingPatch = 0;

    for (int pid = 0; pid < numPatches; ++pid) {
        auto [P_rot, xExtent] = pcaAlignP(patchParams[pid].P);
        patchParams[pid].P = std::move(P_rot);

        double sf = solver.platewidth / xExtent;
        spdlog::info("Patch {} P extent: {:.4f}, scale to platewidth: {:.6f}",
                     pid, xExtent, sf);
        if (sf < globalScale) {
            globalScale = sf;
            constrainingPatch = pid;
        }
    }

    spdlog::info("Global scale: {:.6f} (constrained by patch {} -> platewidth {})",
                 globalScale, constrainingPatch, solver.platewidth);

    // =====================================================================
    // Phase 3: Apply globalScale to V and P, then inverse design
    // =====================================================================
    for (int pid = 0; pid < numPatches; ++pid) {
        Eigen::MatrixXd V_scaled = patchParams[pid].V * globalScale;
        Eigen::MatrixXd P_scaled = patchParams[pid].P * globalScale;
        const Eigen::MatrixXi& F_patch = patchParams[pid].F;
        const int nF_patch = static_cast<int>(F_patch.rows());

        spdlog::info("=== Patch {} inverse design: {} vertices, {} faces ===",
                     pid, V_scaled.rows(), nF_patch);

        // Write param mesh (2D, at platewidth scale)
        {
            Eigen::MatrixXd P_3d = Eigen::MatrixXd::Zero(P_scaled.rows(), 3);
            P_3d.col(0) = P_scaled.col(0);
            P_3d.col(1) = P_scaled.col(1);
            igl::writeOBJ(config.paramDir() + "patch_" + std::to_string(pid) + "_param.obj",
                          P_3d, F_patch);
        }

        // Build geometry-central structures from scaled V and P
        ManifoldSurfaceMesh mesh_p(F_patch);
        VertexPositionGeometry geom_p(mesh_p, V_scaled);
        geom_p.refreshQuantities();

        FaceData<Eigen::Matrix2d> MrInv_p = precomputeMrInv(mesh_p, P_scaled, F_patch);
        std::vector<int> fixedIdx_p = findCenterFaceIndices(P_scaled, F_patch);
        writeCondFile(condDir + "patch_" + std::to_string(pid) + "_bound_center.txt",
                      fixedIdx_p);

        // Inverse design
        InverseDesignProblem problem;
        problem.V         = V_scaled;
        problem.F         = F_patch;
        problem.P         = P_scaled;
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

        // Write target (at globalScale) and proj (rigid-aligned to target)
        igl::writeOBJ(morphDir + "patch_" + std::to_string(pid) + "_target.obj",
                      V_scaled, F_patch);
        igl::writeOBJ(morphDir + "patch_" + std::to_string(pid) + "_proj.obj",
                      result.V_proj, F_patch);

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
