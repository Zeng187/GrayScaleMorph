

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
#include "parameterization.h"
#include "simulation_utils.h"
#include "functions.h"
#include "newton.h"
#include "morphmesh.hpp"
#include "morph_functions.hpp"
#include "output.hpp"
#include "patch_utils.h"

// #define __VERFIY_FORWARD_PREDIT__
#define __VERFIY_INVERSE_DESIGN__

#define __Add_PENALTY__

/// Compute the optimal gauge scale t* for ASAP parameterization.
/// Solves: t* = argmin_t  Σ A_f · (t·λ_f − clamp(t·λ_f, λ_min, λ_max))²
/// After calling, scale P /= t* so that λ → t*·λ moves into the material window.
static double computeGaugeShiftScale(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& P,
    double lambda_min,
    double lambda_max)
{
    const int nF = F.rows();
    Eigen::VectorXd lambda_raw = Eigen::VectorXd::Ones(nF);
    Eigen::VectorXd face_area = Eigen::VectorXd::Zero(nF);

    for (int fi = 0; fi < nF; ++fi) {
        Eigen::Vector3d v0 = V.row(F(fi, 0));
        Eigen::Vector3d v1 = V.row(F(fi, 1));
        Eigen::Vector3d v2 = V.row(F(fi, 2));
        Eigen::Matrix<double, 3, 2> M;
        M.col(0) = v1 - v0;
        M.col(1) = v2 - v0;

        Eigen::Vector2d p0 = P.row(F(fi, 0));
        Eigen::Vector2d p1 = P.row(F(fi, 1));
        Eigen::Vector2d p2 = P.row(F(fi, 2));
        Eigen::Matrix2d Mr;
        Mr.col(0) = p1 - p0;
        Mr.col(1) = p2 - p0;

        const double det_Mr = Mr.determinant();
        face_area(fi) = 0.5 * std::abs(det_Mr);

        if (std::abs(det_Mr) < 1e-16) {
            lambda_raw(fi) = 1.0;
            continue;
        }

        Eigen::Matrix<double, 3, 2> Fg = M * Mr.inverse();
        Eigen::Matrix2d a = Fg.transpose() * Fg;
        lambda_raw(fi) = std::sqrt(std::max(0.0, 0.5 * a.trace()));
    }

    // Initial guess: align geometric mean to window center
    double log_sum = 0.0, area_sum = 0.0;
    for (int fi = 0; fi < nF; ++fi) {
        if (lambda_raw(fi) > 0.0 && face_area(fi) > 0.0) {
            log_sum += face_area(fi) * std::log(lambda_raw(fi));
            area_sum += face_area(fi);
        }
    }
    if (area_sum <= 0.0) return 1.0;

    const double geom_mean = std::exp(log_sum / area_sum);
    if (!(geom_mean > 0.0) || !std::isfinite(geom_mean)) return 1.0;

    double t = std::sqrt(lambda_min * lambda_max) / geom_mean;
    if (!(t > 0.0) || !std::isfinite(t)) return 1.0;

    // Iterative refinement: solve dF/dt = 0 with set partitioning
    for (int iter = 0; iter < 5; ++iter) {
        double num = 0.0, den = 0.0;
        bool any_outside = false;

        for (int fi = 0; fi < nF; ++fi) {
            const double tl = t * lambda_raw(fi);
            const double A = face_area(fi);
            const double l = lambda_raw(fi);

            if (tl < lambda_min) {
                num += A * l * lambda_min;
                den += A * l * l;
                any_outside = true;
            } else if (tl > lambda_max) {
                num += A * l * lambda_max;
                den += A * l * l;
                any_outside = true;
            }
        }

        if (!any_outside || den <= 0.0) break;

        const double t_new = num / den;
        if (!(t_new > 0.0) || !std::isfinite(t_new)) break;
        if (std::abs(t_new - t) < 1e-10 * std::max(1.0, std::abs(t))) {
            t = t_new;
            break;
        }
        t = t_new;
    }

    return t;
}

/// Run the full SGN inverse design pipeline on a single mesh piece.
/// Stores per-face t1, t2 results into global_t1, global_t2 at the given global_face_ids.
/// Also writes per-face metrics (lambda/kappa excess) into global_lam_excess, global_kap_excess.
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

    int nV_p = V_patch.rows();
    int nF_p = F_patch.rows();

    spdlog::info("=== Patch {} inverse design: {} vertices, {} faces ===", patch_id, nV_p, nF_p);

    // Build geometry-central structures for this patch
    ManifoldSurfaceMesh mesh_p(F_patch);
    VertexPositionGeometry geom_p(mesh_p, V_patch);
    geom_p.refreshQuantities();

    // Precompute MrInv from parameterization
    FaceData<Eigen::Matrix2d> MrInv_p = precomputeMrInv(mesh_p, P_patch, F_patch);

    // Find fixed vertex indices for this patch (center face of parameterization)
    std::vector<int> fixedIdx_p = findCenterFaceIndices(P_patch, F_patch);

    // Compute morphing parameters (target lambda, kappa)
    double E = 1.0;
    double nu = 0.5;
    Morphmesh morph_p(V_patch, P_patch, F_patch, E, nu);

    std::vector<bool> bv_flags(nV_p, false);
    std::vector<bool> bf_flags(nF_p, false);
    std::vector<int>  b_ref(nF_p, 0);

    Morphmesh::ComputeMorphophing(geom_p, V_patch, F_patch,
        nV_p, nF_p,
        bv_flags, bf_flags, b_ref, MrInv_p,
        morph_p.lambda_pv_t, morph_p.lambda_pf_t,
        morph_p.kappa_pv_t, morph_p.kappa_pf_t,
        &morph_p.vertex_area_sum);

    Morphmesh::SetMorphophing(
        morph_p.lambda_pv_t, morph_p.lambda_pf_t,
        morph_p.kappa_pv_t, morph_p.kappa_pf_t,
        morph_p.lambda_pv_s, morph_p.lambda_pf_s,
        morph_p.kappa_pv_s, morph_p.kappa_pf_s);

    Eigen::MatrixXd targetV_p = V_patch;

    // Wrap morph params into FaceData/VertexData
    FaceData<bool> is_boundary_face_p(mesh_p);
    FaceData<int> boundary_ref_index_p(mesh_p);
    VertexData<bool> is_boundary_vertex_p(mesh_p);
    VertexData<double> lambda_pv_s(mesh_p, morph_p.lambda_pv_s);
    VertexData<double> kappa_pv_s(mesh_p, morph_p.kappa_pv_s);
    FaceData<double> lambda_pf_s(mesh_p, morph_p.lambda_pf_s);
    FaceData<double> kappa_pf_s(mesh_p, morph_p.kappa_pf_s);

    // SGN inverse design loop
    spdlog::info("Patch {}: Starting SGN inverse design.", patch_id);

    double wP_kap = config.RuntimeSetting.wP_kap;
    double wP_lam = config.RuntimeSetting.wP_lam;
    double penalty_threshold = config.RuntimeSetting.penalty_threshold;
    double betaP = config.RuntimeSetting.betaP;

    int stage_iter = 5;
    int k = 0;

    double wM_kap = config.RuntimeSetting.wM_kap;
    double wM_lam = config.RuntimeSetting.wM_lam;
    double wL_kap = config.RuntimeSetting.wL_kap;
    double wL_lam = config.RuntimeSetting.wL_lam;

    const double wM_kap_init = wM_kap;
    const double wM_lam_init = wM_lam;
    const double wL_kap_init = wL_kap;
    const double wL_lam_init = wL_lam;

    double distance = 0.0;
    double penalty_kap = 0.0;
    double penalty_lam = 0.0;

    Eigen::MatrixXd Vr_p = V_patch;

    while (k < stage_iter)
    {
        // Create joint penalty for OptKap: variable=kappa(per-vertex), fixed=lambda(per-face)
        auto penalty_to_kapp = JointMaterialPenaltyPerV_OptKap(
            geom_p, F_patch, lambda_pf_s, ac.feasible_lamb, ac.feasible_kapp, betaP);

        spdlog::info("Patch {} Stage {}, OptKap start, wP_kap: {:.6f}, wP_lam: {:.6f}.", patch_id, k, wP_kap, wP_lam);
        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geom_p, F_patch, MrInv_p, lambda_pf_s,
            E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
        Vr_p = sparse_gauss_newton_FixLam_OptKap_Penalty(geom_p, targetV_p, Vr_p, MrInv_p,
            lambda_pf_s, kappa_pv_s, adjointFunc_OptKap, penalty_to_kapp, fixedIdx_p,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_kap, wL_kap, wP_kap,
            E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);

        distance = (Vr_p - targetV_p).squaredNorm() / nV_p;
        penalty_kap = compute_candidate_diff(ac.feasible_kapp, kappa_pv_s.toVector(), true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
        spdlog::info("Patch {} Stage {}, OptKap finish - Distance: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                     patch_id, k, distance, penalty_kap, penalty_lam);

        // Create joint penalty for OptLam: variable=lambda(per-face), fixed=kappa(per-vertex)
        auto penalty_to_lamb = JointMaterialPenaltyPerF_OptLam(
            geom_p, F_patch, kappa_pv_s, ac.feasible_lamb, ac.feasible_kapp, betaP);

        spdlog::info("Patch {} Stage {}, OptLam start, wP_kap: {:.6f}, wP_lam: {:.6f}.", patch_id, k, wP_kap, wP_lam);
        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geom_p, F_patch, MrInv_p, kappa_pv_s,
            E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
        Vr_p = sparse_gauss_newton_FixKap_OptLam_Penalty(geom_p, targetV_p, Vr_p, MrInv_p,
            lambda_pf_s, kappa_pv_s, adjointFunc_OptLam, penalty_to_lamb, fixedIdx_p,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_lam, wL_lam, wP_lam,
            E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);

        distance = (Vr_p - targetV_p).squaredNorm() / nV_p;
        penalty_kap = compute_candidate_diff(ac.feasible_kapp, kappa_pv_s.toVector(), true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
        spdlog::info("Patch {} Stage {}, OptLam finish - Distance: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                     patch_id, k, distance, penalty_kap, penalty_lam);

        // Evaluate projected distance
        {
            FaceData<double> kappa_pf_proj(mesh_p);
            FaceData<double> lambda_pf_proj(mesh_p);

            for (Face f : mesh_p.faces()) {
                double sum = 0.0; int cnt = 0;
                for (Vertex v : f.adjacentVertices()) { sum += kappa_pv_s[v]; cnt++; }
                double kap = sum / cnt;
                double lam = lambda_pf_s[f];
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap, lam);
                kappa_pf_proj[f] = ac.feasible_kapp[idx];
                lambda_pf_proj[f] = ac.feasible_lamb[idx];
            }

            auto simFunc_proj = simulationFunction(geom_p, MrInv_p, lambda_pf_proj, kappa_pf_proj,
                E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
            // Start from flat (P_patch, 0) to match physical initial state
            Eigen::MatrixXd Vr_proj = Eigen::MatrixXd::Zero(P_patch.rows(), 3);
            Vr_proj.col(0) = P_patch.col(0);
            Vr_proj.col(1) = P_patch.col(1);
            newton(geom_p, Vr_proj, simFunc_proj,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx_p);
            double dist_proj = (Vr_proj - targetV_p).squaredNorm() / nV_p;

            spdlog::info("Patch {} Stage {}, Projected distance: {:.6f}", patch_id, k, dist_proj);
        }

        k++;
        if (penalty_kap >= penalty_threshold) {
            wP_kap *= 2.0;
        }
        if (penalty_lam >= penalty_threshold) {
            wP_lam *= 2.0;
        }

        wM_kap = std::max(wM_kap * 0.5, wM_kap_init * 1e-3);
        wL_kap = std::max(wL_kap * 0.5, wL_kap_init * 1e-3);
        wM_lam = std::max(wM_lam * 0.5, wM_lam_init * 1e-3);
        wL_lam = std::max(wL_lam * 0.5, wL_lam_init * 1e-3);

        if (penalty_kap < penalty_threshold && penalty_lam < penalty_threshold)
            break;
    }

    // Material projection + forward verification for this patch
    spdlog::info("Patch {}: Material projection + forward verification.", patch_id);

    FaceData<double> kappa_pf_final(mesh_p);
    FaceData<double> lambda_pf_final(mesh_p);
    FaceData<double> t1_pf(mesh_p);
    FaceData<double> t2_pf(mesh_p);

    for (Face f : mesh_p.faces()) {
        double kap_sum = 0.0;
        int cnt = 0;
        for (Vertex v : f.adjacentVertices()) {
            kap_sum += kappa_pv_s[v];
            cnt++;
        }
        double kap_f = kap_sum / cnt;
        double lam_f = lambda_pf_s[f];

        int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap_f, lam_f);
        kappa_pf_final[f] = ac.feasible_kapp[idx];
        lambda_pf_final[f] = ac.feasible_lamb[idx];
        t1_pf[f] = ac.feasible_t_vals[idx].first;
        t2_pf[f] = ac.feasible_t_vals[idx].second;
    }

    // Forward simulation with projected material — start from flat (P_patch, 0)
    auto simFunc_final = simulationFunction(geom_p, MrInv_p, lambda_pf_final, kappa_pf_final,
        E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
    Eigen::MatrixXd Vr_final = Eigen::MatrixXd::Zero(P_patch.rows(), 3);
    Vr_final.col(0) = P_patch.col(0);
    Vr_final.col(1) = P_patch.col(1);
    newton(geom_p, Vr_final, simFunc_final,
        config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, true, fixedIdx_p);

    double dist_final = (Vr_final - targetV_p).squaredNorm() / nV_p;
    spdlog::info("Patch {}: Final projected distance: {:.6f}", patch_id, dist_final);

    // Write per-patch OBJ (projected shape)
    {
        std::string patch_proj_path = morphDir + "patch_" + std::to_string(patch_id) + "_proj.obj";
        igl::writeOBJ(patch_proj_path, Vr_final, F_patch);
        spdlog::info("Patch {}: Projected shape written to: {}", patch_id, patch_proj_path);
    }

    // Map results back to global arrays
    int local_fid = 0;
    for (Face f : mesh_p.faces()) {
        int gfid = global_face_ids[local_fid];
        global_t1[gfid] = t1_pf[f];
        global_t2[gfid] = t2_pf[f];

        double lam_f = lambda_pf_s[f];
        double kap_sum = 0.0; int cnt = 0;
        for (Vertex v : f.adjacentVertices()) { kap_sum += kappa_pv_s[v]; cnt++; }
        double kap_f = kap_sum / cnt;
        double lam_excess = std::max(0.0, std::max(lam_f - ac.range_lam.y, ac.range_lam.x - lam_f));
        double kap_excess = std::max(0.0, std::max(kap_f - ac.range_kap.y, ac.range_kap.x - kap_f));

        global_lam_excess[gfid] = lam_excess;
        global_kap_excess[gfid] = kap_excess;
        local_fid++;
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

    int nF_global = F_global.rows();

    // Load segmentation
    std::vector<int> seg_id = loadSegId(segid_path);
    if ((int)seg_id.size() != nF_global) {
        spdlog::error("seg_id size ({}) != mesh face count ({})", seg_id.size(), nF_global);
        return -1;
    }

    int num_patches = *std::max_element(seg_id.begin(), seg_id.end()) + 1;
    spdlog::info("Per-patch mode: {} patches detected.", num_patches);

    double platewidth = config.RuntimeSetting.Platewidth;

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
        std::string output_mesh_targ_path = config.OutputSetting.OutputPath +
            config.ModelSetting.ModelName + "_targ.obj";
        igl::writeOBJ(output_mesh_targ_path, V_targ, F_global);
        igl::writeOBJ(morphDir + config.ModelSetting.ModelName + "_targ.obj", V_targ, F_global);
    }

    // Process each patch
    for (int pid = 0; pid < num_patches; ++pid) {
        PatchData patch = extractPatch(V_scaled, F_global, seg_id, pid);
        if (patch.F.rows() == 0) {
            spdlog::warn("Patch {} has 0 faces, skipping.", pid);
            continue;
        }
        spdlog::info("Patch {}: {} faces, {} vertices", pid, patch.F.rows(), patch.V.rows());

        // Parameterize the patch
        Eigen::MatrixXi F_p = patch.F;
        int nF_orig = F_p.rows();

        // Fill holes for parameterization (boundary patches need this)
        std::vector<int> boundary = fillInHoles(patch.V, F_p);
        Eigen::MatrixXd P_p = tutteEmbedding(patch.V, F_p, boundary);

        LocalGlobalSolver solver(patch.V, F_p);
        solver.solve(P_p, 1.0 / ac.range_lam.y, 1.0 / ac.range_lam.x);
        centerAndRotate(patch.V, P_p);

        // Restore original face count (remove filled-in faces)
        F_p.conservativeResize(nF_orig, 3);

        // Gauge shift: position λ distribution into material window [λ_min, λ_max]
        {
            const double t = computeGaugeShiftScale(
                patch.V, F_p, P_p, ac.range_lam.x, ac.range_lam.y);
            if (t > 0.0 && std::isfinite(t) && std::abs(t - 1.0) > 1e-12) {
                P_p /= t;
                spdlog::info("Patch {}: gauge shift t={:.6f}, P scaled by {:.6f}", pid, t, 1.0 / t);
            }
        }

        // Second scaling: fit parameterization to platewidth
        double s2 = platewidth / (P_p.colwise().maxCoeff() - P_p.colwise().minCoeff()).maxCoeff();
        Eigen::MatrixXd V_p = patch.V * s2;
        P_p *= s2;

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
        std::string output_material_path = config.OutputSetting.OutputPath +
            config.ModelSetting.ModelName + "_material.txt";
        std::string design_material_path = designDir + config.ModelSetting.ModelName + "_material.txt";

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
    ActiveComposite ac(config.ResourceSetting.MaterialPath);
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    // Create output directories
    std::string morphDir  = config.OutputSetting.MorphPath + config.ModelSetting.ModelName + "/";
    std::string designDir = config.OutputSetting.DesignPath + config.ModelSetting.ModelName + "/";
    std::string metricsDir = config.OutputSetting.MetricsPath + config.ModelSetting.ModelName + "/";
    std::filesystem::create_directories(config.OutputSetting.OutputPath);
    std::filesystem::create_directories(morphDir);
    std::filesystem::create_directories(designDir);
    std::filesystem::create_directories(metricsDir);

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    spdlog::info("program start:");

	///***************************************** Load Mesh *****************************************///

    std::string input_mesh_path = config.ModelSetting.InputPath +  config.ModelSetting.ModelName + config.ModelSetting.Postfix;
    if (!igl::readOBJ(input_mesh_path, V, F)) {
        spdlog::error("Error: Could not read output_mesh.obj\n");
        return -1;
    }

    size_t nV = V.rows();
    size_t nF = F.rows();
    spdlog::info("Read meshes with {0} vertices and {1} faces.",nV,nF);


    while (nF < config.RuntimeSetting.nFmin)
    {
        Eigen::MatrixXd tempV = V;
        Eigen::MatrixXi tempF = F;
        igl::loop(tempV, tempF, V, F);
        nV = V.rows();
        nF = F.rows();
    }

    ///***************************************** Check for per-patch mode *****************************************///

    std::string segDir = config.segmentDir(config.ModelSetting.ModelName);
    std::string segid_path = segDir + "seg_id.txt";
    if (std::filesystem::exists(segid_path)) {
        spdlog::info("Segmentation file found: {}. Running per-patch inverse design.", segid_path);
        int ret = runPerPatchMode(config, ac, V, F, segid_path, morphDir, designDir, metricsDir);
        spdlog::info("program finish.");
        return ret;
    }

    // Fallback: try plain model name (backward compatibility with old directory layout)
    if (!config.ResourceSetting.DistortionMethod.empty()) {
        std::string segid_path_fallback = config.ResourceSetting.SegmentPath + config.ModelSetting.ModelName + "/seg_id.txt";
        if (std::filesystem::exists(segid_path_fallback)) {
            spdlog::info("Segmentation file found at fallback path: {}. Running per-patch inverse design.", segid_path_fallback);
            int ret = runPerPatchMode(config, ac, V, F, segid_path_fallback, morphDir, designDir, metricsDir);
            spdlog::info("program finish.");
            return ret;
        }
    }

    spdlog::info("No segmentation file found at: {}. Running whole-mesh mode.", segid_path);

    ///***************************************** Whole-mesh mode (original pipeline) *****************************************///

    double scaleFactor = 1.0;
    double scaleFactor_1 = config.RuntimeSetting.Platewidth / (V.colwise().maxCoeff() - V.colwise().minCoeff()).maxCoeff();
    scaleFactor *= scaleFactor_1;
    V *= scaleFactor_1;

    ManifoldSurfaceMesh mesh(F);
    VertexPositionGeometry geometry(mesh, V);
    geometry.refreshQuantities();

    ///***************************************** Parameterization *****************************************///
    spdlog::info("Step 1: Parameterization.");
    Eigen::MatrixXd P = parameterization(V, F, ac.range_lam.x, ac.range_lam.y, 0);

    // Gauge shift: position λ distribution into material window [λ_min, λ_max]
    {
        const double t = computeGaugeShiftScale(V, F, P, ac.range_lam.x, ac.range_lam.y);
        if (t > 0.0 && std::isfinite(t) && std::abs(t - 1.0) > 1e-12) {
            P /= t;
            spdlog::info("Whole mesh: gauge shift t={:.6f}, P scaled by {:.6f}", t, 1.0 / t);
        }
    }

    double scaleFactor_2 = config.RuntimeSetting.Platewidth / (P.colwise().maxCoeff() - P.colwise().minCoeff()).maxCoeff();
    scaleFactor *= scaleFactor_2;
    V *= scaleFactor_2;
    P *= scaleFactor_2;
    geometry.inputVertexPositions *= scaleFactor_2;
    geometry.refreshQuantities();


    std::vector<bool> boundary_vertex_flags(nV, false);
    std::vector<bool> boundary_face_flags(nF, false);
    std::vector<int> boundary_ref_indices(nF, 0);

    // Output 2D parameterized mesh (flat domain in platewidth scale) for Abaqus simulation
    {
        Eigen::MatrixXd P_3d = Eigen::MatrixXd::Zero(P.rows(), 3);
        P_3d.col(0) = P.col(0);
        P_3d.col(1) = P.col(1);
        igl::writeOBJ(morphDir + config.ModelSetting.ModelName + "_param.obj", P_3d, F);
        spdlog::info("Parameterized mesh written to: {}", morphDir + config.ModelSetting.ModelName + "_param.obj");
    }

    ///***************************************** Material Settings *****************************************///

    spdlog::info("Step 2: Material Settings.");

    Eigen::MatrixXd targetV = V;

    FaceData<Eigen::MatrixXd> M = precomputeM(mesh, V, F);
    FaceData<Eigen::Matrix2d> MrInv = precomputeMrInv(mesh, P, F);

    std::vector<int> fixedVertexIdx = findCenterVertexIndices(P, F);
    std::vector<int> fixedIdx = findCenterFaceIndices(P, F);  // 9 DOF indices: {3*v, 3*v+1, 3*v+2} for each of 3 center face vertices

    double E = 1.0;
    double nu = 0.5;
    Morphmesh morph_mesh(V, P, F, E, nu);
    Morphmesh::ComputeMorphophing(geometry, V, F, nV, nF,boundary_vertex_flags,boundary_face_flags,boundary_ref_indices,
        MrInv, morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t,morph_mesh.kappa_pv_t,morph_mesh.kappa_pf_t, &morph_mesh.vertex_area_sum);
    Morphmesh::SetMorphophing(morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t,
        morph_mesh.kappa_pv_t,morph_mesh.kappa_pf_t,
        morph_mesh.lambda_pv_s, morph_mesh.lambda_pf_s,
        morph_mesh.kappa_pv_s, morph_mesh.kappa_pf_s);
	// Morphmesh::RestrictRange(morph_mesh.lambda_pv_s, ac.range_lam.x, ac.range_lam.y);
	// Morphmesh::RestrictRange(morph_mesh.kappa_pv_s, ac.range_kap.x, ac.range_kap.y);
    // Morphmesh::RestrictRange(morph_mesh.lambda_pf_s, ac.range_lam.x, ac.range_lam.y);
    // Morphmesh::RestrictRange(morph_mesh.kappa_pf_s, ac.range_kap.x, ac.range_kap.y);


    auto V_targ= V;
    V_targ *= 1.0 / scaleFactor;
    std::string output_mesh_targ_path = config.OutputSetting.OutputPath +
        config.ModelSetting.ModelName + "_targ" + ".obj";
    igl::writeOBJ(output_mesh_targ_path, V_targ, F);
    igl::writeOBJ(morphDir + config.ModelSetting.ModelName + "_targ.obj", V_targ, F);


    FaceData<bool> is_boundary_face(mesh);
    FaceData<int> boundary_ref_index(mesh);
    VertexData<bool> is_boundary_vertex(mesh);
    VertexData<double> lambda_pv_s(mesh, morph_mesh.lambda_pv_s);
    VertexData<double> kappa_pv_s(mesh, morph_mesh.kappa_pv_s);
    FaceData<double> lambda_pf_s(mesh, morph_mesh.lambda_pf_s);
    FaceData<double> kappa_pf_s(mesh, morph_mesh.kappa_pf_s);

    ///***************************************** Forward Predit *****************************************///

    // spdlog::info("Step 3: Forward Predit.");


    auto V_pred = V, Vr = V;

    // auto simul_func_2 = simulationFunction(geometry,
    //     MrInv,
    //     lambda_pv_s,
    //     kappa_pv_s,
	// 	E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b
    //     );

    // Vr = V;
    // newton(geometry, Vr, simul_func_2, config.RuntimeSetting.MaxIter,
    //     config.RuntimeSetting.epsilon, true, fixedIdx);

    // V_pred = Vr;
    // V_pred *= 1.0 / scaleFactor;
    // std::string output_mesh_pred_path_2 = config.OutputSetting.OutputPath +
    //     config.ModelSetting.ModelName + "_pred" + ".obj";
    // igl::writeOBJ(output_mesh_pred_path_2, V_pred, F);


    // ///***************************************** Inverse Design *****************************************///

    spdlog::info("Step 4: Inverse Design.");

    double wP_kap = config.RuntimeSetting.wP_kap;
    double wP_lam = config.RuntimeSetting.wP_lam;
    double penalty_threshold = config.RuntimeSetting.penalty_threshold;
    double betaP = config.RuntimeSetting.betaP;
    int stage_iter = 5;
    int k = 0;

    double wM_kap = config.RuntimeSetting.wM_kap;
    double wM_lam = config.RuntimeSetting.wM_lam;
    double wL_kap = config.RuntimeSetting.wL_kap;
    double wL_lam = config.RuntimeSetting.wL_lam;

    const double wM_kap_init = wM_kap;
    const double wM_lam_init = wM_lam;
    const double wL_kap_init = wL_kap;
    const double wL_lam_init = wL_lam;

    double distance = 0.0;
    double penalty_kap = 0.0;
    double penalty_lam = 0.0;
    while(k < stage_iter)
    {
        // Create joint penalty for OptKap: variable=kappa(per-vertex), fixed=lambda(per-face)
        auto penalty_to_kapp = JointMaterialPenaltyPerV_OptKap(
            geometry, F, lambda_pf_s, ac.feasible_lamb, ac.feasible_kapp, betaP);

        spdlog::info("Stage {}, OptKap start, wP_kap: {:.6f}, wP_lam: {:.6f}.", k, wP_kap, wP_lam);
        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
        Vr = sparse_gauss_newton_FixLam_OptKap_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s, adjointFunc_OptKap, penalty_to_kapp, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_kap, wL_kap, wP_kap,
            E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b);

        // Compute and output distance and penalties after OptKap
        distance = (Vr - targetV).squaredNorm() / nV;
        penalty_kap = compute_candidate_diff(ac.feasible_kapp,kappa_pv_s.toVector(),true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb,lambda_pf_s.toVector(),true);
        spdlog::info("Stage {}, OptKap finish - Distance: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                     k, distance, penalty_kap, penalty_lam);

        // Create joint penalty for OptLam: variable=lambda(per-face), fixed=kappa(per-vertex)
        auto penalty_to_lamb = JointMaterialPenaltyPerF_OptLam(
            geometry, F, kappa_pv_s, ac.feasible_lamb, ac.feasible_kapp, betaP);

        spdlog::info("Stage {}, OptLam start, wP_kap: {:.6f}, wP_lam: {:.6f}.", k, wP_kap, wP_lam);
        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pv_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
        Vr = sparse_gauss_newton_FixKap_OptLam_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pv_s, adjointFunc_OptLam, penalty_to_lamb, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_lam, wL_lam, wP_lam,
            E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b);

        // Compute and output distance and penalties after OptLam
        distance = (Vr - targetV).squaredNorm() / nV;
        penalty_kap = compute_candidate_diff(ac.feasible_kapp,kappa_pv_s.toVector(),true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb,lambda_pf_s.toVector(),true);
        spdlog::info("Stage {}, OptLam finish- Distance: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                     k, distance, penalty_kap, penalty_lam);

        // Evaluate distance after jointly projecting kappa and lambda to the same feasible index
        {
            FaceData<double> kappa_pf_proj(mesh);
            FaceData<double> lambda_pf_proj(mesh);

            for (Face f : mesh.faces()) {
                double sum = 0.0; int cnt = 0;
                for (Vertex v : f.adjacentVertices()) { sum += kappa_pv_s[v]; cnt++; }
                double kap = sum / cnt;
                double lam = lambda_pf_s[f];
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap, lam);
                kappa_pf_proj[f] = ac.feasible_kapp[idx];
                lambda_pf_proj[f] = ac.feasible_lamb[idx];
            }

            auto simFunc_proj = simulationFunction(geometry, MrInv, lambda_pf_proj, kappa_pf_proj,
                E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
            // Start from flat (P, 0) to match physical initial state
            Eigen::MatrixXd Vr_proj = Eigen::MatrixXd::Zero(P.rows(), 3);
            Vr_proj.col(0) = P.col(0);
            Vr_proj.col(1) = P.col(1);
            newton(geometry, Vr_proj, simFunc_proj,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx);
            double dist_proj = (Vr_proj - targetV).squaredNorm() / nV;

            spdlog::info("Stage {}, Projected distance: {:.6f}", k, dist_proj);
        }

        k++;
        if (penalty_kap >= penalty_threshold) {
            wP_kap *= 2.0;
        }
        if (penalty_lam >= penalty_threshold) {
            wP_lam *= 2.0;
        }

        wM_kap = std::max(wM_kap * 0.5, wM_kap_init * 1e-3);
        wL_kap = std::max(wL_kap * 0.5, wL_kap_init * 1e-3);
        wM_lam = std::max(wM_lam * 0.5, wM_lam_init * 1e-3);
        wL_lam = std::max(wL_lam * 0.5, wL_lam_init * 1e-3);

        if(penalty_kap < penalty_threshold && penalty_lam < penalty_threshold)
            break;

        // wM_kap *=0.1;
        // wM_lam *=0.1;
        // wL_kap *=0.1;
        // wL_lam *=0.1;
    }


    ///***************************************** Material Projection + Forward Verification *****************************************///

    spdlog::info("Step 5: Material Projection + Forward Verification.");

    // Step 5a: Per-face joint projection to feasible material set
    FaceData<double> kappa_pf_final(mesh);
    FaceData<double> lambda_pf_final(mesh);
    FaceData<double> t1_pf(mesh);
    FaceData<double> t2_pf(mesh);

    for (Face f : mesh.faces()) {
        double kap_sum = 0.0;
        int cnt = 0;
        for (Vertex v : f.adjacentVertices()) {
            kap_sum += kappa_pv_s[v];
            cnt++;
        }
        double kap_f = kap_sum / cnt;
        double lam_f = lambda_pf_s[f];

        int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap_f, lam_f);
        kappa_pf_final[f] = ac.feasible_kapp[idx];
        lambda_pf_final[f] = ac.feasible_lamb[idx];
        t1_pf[f] = ac.feasible_t_vals[idx].first;
        t2_pf[f] = ac.feasible_t_vals[idx].second;
    }

    // Step 5b: Forward simulation with projected material (start from flat)
    auto simFunc_final = simulationFunction(geometry, MrInv, lambda_pf_final, kappa_pf_final,
        E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
    // Start from flat (P, 0) to match physical initial state (Abaqus starts from flat sheet)
    Eigen::MatrixXd Vr_final = Eigen::MatrixXd::Zero(P.rows(), 3);
    Vr_final.col(0) = P.col(0);
    Vr_final.col(1) = P.col(1);
    newton(geometry, Vr_final, simFunc_final,
        config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, true, fixedIdx);

    double dist_final = (Vr_final - targetV).squaredNorm() / nV;
    spdlog::info("Final projected distance: {:.6f}", dist_final);

    // Step 5c: Output projected shape
    spdlog::info("scaleFactor = {} (s1={}, s2={})", scaleFactor, scaleFactor_1, scaleFactor_2);

    // Output in platewidth scale (same coordinate system as _param.obj and Abaqus)
    igl::writeOBJ(morphDir + config.ModelSetting.ModelName + "_proj_pw.obj", Vr_final, F);
    spdlog::info("Projected shape (platewidth scale) written to: {}",
        morphDir + config.ModelSetting.ModelName + "_proj_pw.obj");

    Eigen::MatrixXd V_proj = Vr_final;
    V_proj *= 1.0 / scaleFactor;
    std::string output_mesh_proj_path = config.OutputSetting.OutputPath +
        config.ModelSetting.ModelName + "_proj" + ".obj";
    igl::writeOBJ(output_mesh_proj_path, V_proj, F);
    igl::writeOBJ(morphDir + config.ModelSetting.ModelName + "_proj.obj", V_proj, F);
    spdlog::info("Projected shape written to: {}", output_mesh_proj_path);

    // Step 5d: Output per-face material file
    std::string output_material_path = config.OutputSetting.OutputPath +
        config.ModelSetting.ModelName + "_material" + ".txt";
    std::string design_material_path = designDir + config.ModelSetting.ModelName + "_material.txt";
    {
        auto writeMaterial = [&](const std::string& path) {
            std::ofstream ofs(path);
            ofs << "# face_id  t1  t2\n";
            int fid = 0;
            for (Face f : mesh.faces()) {
                ofs << fid << "  " << t1_pf[f] << "  " << t2_pf[f] << "\n";
                fid++;
            }
        };
        writeMaterial(output_material_path);
        writeMaterial(design_material_path);
        spdlog::info("Material file written to: {} and {}", output_material_path, design_material_path);
    }

    // Step 5e: Output inverse design shape
    auto V_inv = Vr;
    V_inv *= 1.0 / scaleFactor;
    std::string output_mesh_inv_path = config.OutputSetting.OutputPath +
        config.ModelSetting.ModelName + "_inv" + ".obj";
    igl::writeOBJ(output_mesh_inv_path, V_inv, F);
    igl::writeOBJ(morphDir + config.ModelSetting.ModelName + "_inv.obj", V_inv, F);

    // Step 5f: Output per-face metrics for warm-start
    {
        std::string metrics_path = metricsDir + "metrics.txt";
        std::ofstream ofs(metrics_path);
        ofs << "# face_id  lambda_excess  kappa_excess\n";
        int fid = 0;
        for (Face f : mesh.faces()) {
            double lam_f = lambda_pf_s[f];
            double kap_sum = 0.0; int cnt = 0;
            for (Vertex v : f.adjacentVertices()) { kap_sum += kappa_pv_s[v]; cnt++; }
            double kap_f = kap_sum / cnt;

            double lam_excess = std::max(0.0, std::max(lam_f - ac.range_lam.y, ac.range_lam.x - lam_f));
            double kap_excess = std::max(0.0, std::max(kap_f - ac.range_kap.y, ac.range_kap.x - kap_f));

            ofs << fid << "  " << lam_excess << "  " << kap_excess << "\n";
            fid++;
        }
        spdlog::info("Metrics written to: {}", metrics_path);
    }



    ///***************************************** View by Imgui *****************************************///
    //igl::opengl::glfw::Viewer viewer;
    //viewer.data().set_mesh(V_pred, F);
    ////viewer.data().set_colors(C);
    //viewer.data().show_lines = true;
    //viewer.launch();

    spdlog::info("program finish.");

}
