

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
// #include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/loop.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#ifdef _WIN32
#include <io.h>
#endif

#include <spdlog/spdlog.h>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include"config.hpp"
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

// #define __VERFIY_FORWARD_PREDIT__
#define __VERFIY_INVERSE_DESIGN__

#define __Add_PENALTY__

/// Thin wrapper: build geometry-central objects for a patch, run the shared
/// inverse design core, write per-patch OBJ, and scatter results to global arrays.
static void runInverseDesignOnPatch(
    const Eigen::MatrixXd& V_patch,   // patch vertices (already scaled to platewidth)
    const Eigen::MatrixXi& F_patch,   // patch faces
    const Eigen::MatrixXd& P_patch,   // patch parameterization (already scaled)
    const std::vector<int>& global_face_ids, // mapping from patch face index to global face index
    const Config& config,
    const ActiveComposite& ac,
    int patch_id,
    // Output arrays (indexed by global face id)
    Eigen::VectorXd& global_t1,
    Eigen::VectorXd& global_t2,
    Eigen::VectorXd& global_lam_excess,
    Eigen::VectorXd& global_kap_excess,
    // Output directories for per-patch OBJs
    const std::string& morphDir)
{
    using namespace geometrycentral;
    using namespace geometrycentral::surface;
    const auto& solver = config.solver;

    const int nF_p = static_cast<int>(F_patch.rows());
    spdlog::info("=== Patch {} inverse design: {} vertices, {} faces ===",
                 patch_id, V_patch.rows(), nF_p);

    // Build geometry-central structures for this patch
    ManifoldSurfaceMesh mesh_p(F_patch);
    VertexPositionGeometry geom_p(mesh_p, V_patch);
    geom_p.refreshQuantities();

    FaceData<Eigen::Matrix2d> MrInv_p = precomputeMrInv(mesh_p, P_patch, F_patch);
    std::vector<int> fixedIdx_p = findCenterFaceIndices(P_patch, F_patch);

    // Populate InverseDesignProblem and delegate to the shared core
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

    // Write per-patch OBJ (projected shape)
    {
        std::string patch_proj_path = morphDir + "patch_" + std::to_string(patch_id) + "_proj.obj";
        igl::writeOBJ(patch_proj_path, result.V_proj, F_patch);
        spdlog::info("Patch {}: Projected shape written to: {}", patch_id, patch_proj_path);
    }

    // Map results back to global arrays
    for (int local_fid = 0; local_fid < nF_p; ++local_fid) {
        int gfid = global_face_ids[local_fid];
        global_t1[gfid]          = result.t1[local_fid];
        global_t2[gfid]          = result.t2[local_fid];
        global_lam_excess[gfid]  = result.lam_excess[local_fid];
        global_kap_excess[gfid]  = result.kap_excess[local_fid];
    }

    spdlog::info("=== Patch {} inverse design complete ===", patch_id);
}


/// Per-patch inverse design mode: loads segmentation, runs inverse design on each patch,
/// merges results back to global mesh for output.
static int runPerPatchMode(
    const Config& config,
    ActiveComposite& ac,
    const Eigen::MatrixXd& V_global,
    const Eigen::MatrixXi& F_global,
    const std::string& segid_path,
    const std::string& morphDir,
    const std::string& designDir,
    const std::string& metricsDir)
{
    using namespace geometrycentral;
    using namespace geometrycentral::surface;
    const auto& solver = config.solver;

    int nF_global = F_global.rows();

    // Load segmentation
    std::vector<int> seg_id = loadSegId(segid_path);
    if ((int)seg_id.size() != nF_global) {
        spdlog::error("seg_id size ({}) != mesh face count ({})", seg_id.size(), nF_global);
        return -1;
    }

    int num_patches = *std::max_element(seg_id.begin(), seg_id.end()) + 1;
    spdlog::info("Per-patch mode: {} patches detected.", num_patches);

    double platewidth = solver.platewidth;

    // Scale global mesh to platewidth (same as whole-mesh mode)
    Eigen::MatrixXd V_scaled = V_global;
    double scaleFactor = platewidth / (V_scaled.colwise().maxCoeff() - V_scaled.colwise().minCoeff()).maxCoeff();
    V_scaled *= scaleFactor;

    // Global output arrays
    Eigen::VectorXd global_t1 = Eigen::VectorXd::Zero(nF_global);
    Eigen::VectorXd global_t2 = Eigen::VectorXd::Zero(nF_global);
    Eigen::VectorXd global_lam_excess = Eigen::VectorXd::Zero(nF_global);
    Eigen::VectorXd global_kap_excess = Eigen::VectorXd::Zero(nF_global);

    // Write target mesh
    {
        Eigen::MatrixXd V_targ = V_scaled * (1.0 / scaleFactor);
        std::string output_mesh_targ_path = config.output.output_path +
            config.model.name + "_targ.obj";
        igl::writeOBJ(output_mesh_targ_path, V_targ, F_global);
        igl::writeOBJ(morphDir + config.model.name + "_targ.obj", V_targ, F_global);
    }

    // Process each patch
    for (int pid = 0; pid < num_patches; ++pid) {
        PatchData patch = extractPatch(V_scaled, F_global, seg_id, pid);
        if (patch.F.rows() == 0) {
            spdlog::warn("Patch {} has 0 faces, skipping.", pid);
            continue;
        }
        spdlog::info("Patch {}: {} faces, {} vertices", pid, patch.F.rows(), patch.V.rows());

        // Parameterize the patch (unified pipeline)
        ParameterizeResult param = parameterizeMesh(
            patch.V, patch.F, ac.range_lam.x, ac.range_lam.y, platewidth);
        Eigen::MatrixXd  V_p = std::move(param.V);
        Eigen::MatrixXi  F_p = std::move(param.F);
        Eigen::MatrixXd  P_p = std::move(param.P);

        // Output per-patch parameterized mesh for Abaqus
        {
            Eigen::MatrixXd P_3d = Eigen::MatrixXd::Zero(P_p.rows(), 3);
            P_3d.col(0) = P_p.col(0);
            P_3d.col(1) = P_p.col(1);
            std::string param_path = morphDir + "patch_" + std::to_string(pid) + "_param.obj";
            igl::writeOBJ(param_path, P_3d, F_p);
            spdlog::info("Patch {} param mesh written to: {}", pid, param_path);
        }

        // Run inverse design on this patch
        runInverseDesignOnPatch(
            V_p, F_p, P_p,
            patch.global_face_ids,
            config, ac, pid,
            global_t1, global_t2,
            global_lam_excess, global_kap_excess,
            morphDir);
    }

    // Output merged material file
    {
        std::string output_material_path = config.output.output_path +
            config.model.name + "_material.txt";
        std::string design_material_path = designDir + config.model.name + "_material.txt";

        auto writeMaterial = [&](const std::string& path) {
            std::ofstream ofs(path);
            ofs << "# face_id  t1  t2\n";
            for (int fid = 0; fid < nF_global; ++fid) {
                ofs << fid << "  " << global_t1[fid] << "  " << global_t2[fid] << "\n";
            }
        };
        writeMaterial(output_material_path);
        writeMaterial(design_material_path);
        spdlog::info("Merged material file written to: {} and {}", output_material_path, design_material_path);
    }

    // Output per-face metrics for warm-start
    {
        std::string metrics_path = metricsDir + "metrics.txt";
        std::ofstream ofs(metrics_path);
        ofs << "# face_id  lambda_excess  kappa_excess\n";
        for (int fid = 0; fid < nF_global; ++fid) {
            ofs << fid << "  " << global_lam_excess[fid] << "  " << global_kap_excess[fid] << "\n";
        }
        spdlog::info("Merged metrics written to: {}", metrics_path);
    }

    return 0;
}


int main(int argc, char* argv[])
{


    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    Config config("cfg.json");
    const auto& model   = config.model;
    const auto& output  = config.output;
    const auto& solver  = config.solver;

    ActiveComposite ac(config.material.curves_path);
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    // Create output directories
    std::string morphDir   = config.morphDir();
    std::string designDir  = config.designDir();
    std::string metricsDir = output.output_path + model.name + "/";
    std::filesystem::create_directories(output.output_path);
    std::filesystem::create_directories(morphDir);
    std::filesystem::create_directories(designDir);
    std::filesystem::create_directories(metricsDir);

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    spdlog::info("program start:");

	///***************************************** Load Mesh *****************************************///

    std::string input_mesh_path = model.mesh_path;
    if (!igl::readOBJ(input_mesh_path, V, F)) {
        spdlog::error("Error: Could not read output_mesh.obj\n");
        return -1;
    }

    size_t nV = V.rows();
    size_t nF = F.rows();
    spdlog::info("Read meshes with {0} vertices and {1} faces.",nV,nF);


    while (nF < solver.nf_min)
    {
        Eigen::MatrixXd tempV = V;
        Eigen::MatrixXi tempF = F;
        igl::loop(tempV, tempF, V, F);
        nV = V.rows();
        nF = F.rows();
    }

    ///***************************************** Check for per-patch mode *****************************************///

    std::string segDir = config.segmentDir(model.name);
    std::string segid_path = segDir + "seg_id.txt";
    if (std::filesystem::exists(segid_path)) {
        spdlog::info("Segmentation file found: {}. Running per-patch inverse design.", segid_path);
        int ret = runPerPatchMode(config, ac, V, F, segid_path, morphDir, designDir, metricsDir);
        spdlog::info("program finish.");
        return ret;
    }

    // Fallback: try plain model name (backward compatibility with old directory layout)
    if (!config.segment.method.empty()) {
        std::string segid_path_fallback = config.segment.path + model.name + "/seg_id.txt";
        if (std::filesystem::exists(segid_path_fallback)) {
            spdlog::info("Segmentation file found at fallback path: {}. Running per-patch inverse design.", segid_path_fallback);
            int ret = runPerPatchMode(config, ac, V, F, segid_path_fallback, morphDir, designDir, metricsDir);
            spdlog::info("program finish.");
            return ret;
        }
    }

    spdlog::info("No segmentation file found at: {}. Running whole-mesh mode.", segid_path);

    ///***************************************** Whole-mesh mode *****************************************///

    double scaleFactor = 1.0;
    double scaleFactor_1 = solver.platewidth / (V.colwise().maxCoeff() - V.colwise().minCoeff()).maxCoeff();
    scaleFactor *= scaleFactor_1;
    V *= scaleFactor_1;

    ///***************************************** Parameterization *****************************************///
    spdlog::info("Step 1: Parameterization.");

    ParameterizeResult param = parameterizeMesh(
        V, F, ac.range_lam.x, ac.range_lam.y, solver.platewidth);

    double scaleFactor_2 = param.scaleFactor;
    scaleFactor *= scaleFactor_2;
    V = std::move(param.V);
    F = std::move(param.F);
    Eigen::MatrixXd P = std::move(param.P);

    // Ownership of geometry-central objects: keep unique_ptrs alive,
    // bind references for downstream code that uses mesh/geometry directly.
    auto meshPtr     = std::move(param.mesh);
    auto geometryPtr = std::move(param.geometry);
    ManifoldSurfaceMesh&    mesh     = *meshPtr;
    VertexPositionGeometry& geometry = *geometryPtr;

    FaceData<Eigen::Matrix2d> MrInv = std::move(param.MrInv);
    std::vector<int> fixedIdx       = std::move(param.fixedIdx);

    // Update vertex/face counts after possible face-count changes
    nV = V.rows();
    nF = F.rows();

    // Output 2D parameterized mesh (flat domain in platewidth scale) for Abaqus simulation
    {
        Eigen::MatrixXd P_3d = Eigen::MatrixXd::Zero(P.rows(), 3);
        P_3d.col(0) = P.col(0);
        P_3d.col(1) = P.col(1);
        igl::writeOBJ(morphDir + model.name + "_param.obj", P_3d, F);
        spdlog::info("Parameterized mesh written to: {}", morphDir + model.name + "_param.obj");
    }

    // Output target mesh (in original scale)
    auto V_targ = V;
    V_targ *= 1.0 / scaleFactor;
    std::string output_mesh_targ_path = output.output_path +
        model.name + "_targ.obj";
    igl::writeOBJ(output_mesh_targ_path, V_targ, F);
    igl::writeOBJ(morphDir + model.name + "_targ.obj", V_targ, F);

    ///***************************************** Inverse Design *****************************************///

    spdlog::info("Step 2: Inverse Design.");

    InverseDesignProblem whole_problem;
    whole_problem.V         = V;
    whole_problem.F         = F;
    whole_problem.P         = P;
    whole_problem.mesh      = &mesh;
    whole_problem.geometry  = &geometry;
    whole_problem.MrInv     = MrInv;
    whole_problem.fixedIdx  = fixedIdx;
    whole_problem.ac        = &ac;
    whole_problem.max_iter          = solver.max_iter;
    whole_problem.epsilon           = solver.epsilon;
    whole_problem.w_s               = solver.w_s;
    whole_problem.w_b               = solver.w_b;
    whole_problem.wM_kap            = solver.wM_kap;
    whole_problem.wL_kap            = solver.wL_kap;
    whole_problem.wM_lam            = solver.wM_lam;
    whole_problem.wL_lam            = solver.wL_lam;
    whole_problem.wP_kap            = solver.wP_kap;
    whole_problem.wP_lam            = solver.wP_lam;
    whole_problem.penalty_threshold = solver.penalty_threshold;
    whole_problem.betaP             = solver.betaP;
    whole_problem.patch_id          = -1;

    InverseDesignResult whole_result = runInverseDesign(whole_problem);

    ///***************************************** Output Results *****************************************///

    spdlog::info("Step 3: Writing outputs.");
    spdlog::info("scaleFactor = {} (s1={}, s2={})", scaleFactor, scaleFactor_1, scaleFactor_2);

    // Output projected shape in platewidth scale (same coordinate system as _param.obj and Abaqus)
    igl::writeOBJ(morphDir + model.name + "_proj_pw.obj", whole_result.V_proj, F);
    spdlog::info("Projected shape (platewidth scale) written to: {}",
        morphDir + model.name + "_proj_pw.obj");

    // Output projected shape in original scale
    Eigen::MatrixXd V_proj = whole_result.V_proj;
    V_proj *= 1.0 / scaleFactor;
    std::string output_mesh_proj_path = output.output_path +
        model.name + "_proj.obj";
    igl::writeOBJ(output_mesh_proj_path, V_proj, F);
    igl::writeOBJ(morphDir + model.name + "_proj.obj", V_proj, F);
    spdlog::info("Projected shape written to: {}", output_mesh_proj_path);

    // Output per-face material file
    std::string output_material_path = output.output_path +
        model.name + "_material.txt";
    std::string design_material_path = designDir + model.name + "_material.txt";
    {
        auto writeMaterial = [&](const std::string& path) {
            std::ofstream ofs(path);
            ofs << "# face_id  t1  t2\n";
            for (int fid = 0; fid < static_cast<int>(nF); ++fid) {
                ofs << fid << "  " << whole_result.t1[fid] << "  " << whole_result.t2[fid] << "\n";
            }
        };
        writeMaterial(output_material_path);
        writeMaterial(design_material_path);
        spdlog::info("Material file written to: {} and {}", output_material_path, design_material_path);
    }

    // Output inverse design shape in original scale
    Eigen::MatrixXd V_inv = whole_result.V_inv;
    V_inv *= 1.0 / scaleFactor;
    std::string output_mesh_inv_path = output.output_path +
        model.name + "_inv.obj";
    igl::writeOBJ(output_mesh_inv_path, V_inv, F);
    igl::writeOBJ(morphDir + model.name + "_inv.obj", V_inv, F);

    // Output per-face metrics for warm-start
    {
        std::string metrics_path = metricsDir + "metrics.txt";
        std::ofstream ofs(metrics_path);
        ofs << "# face_id  lambda_excess  kappa_excess\n";
        for (int fid = 0; fid < static_cast<int>(nF); ++fid) {
            ofs << fid << "  " << whole_result.lam_excess[fid] << "  " << whole_result.kap_excess[fid] << "\n";
        }
        spdlog::info("Metrics written to: {}", metrics_path);
    }

    spdlog::info("program finish.");

}
