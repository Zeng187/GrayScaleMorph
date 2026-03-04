

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
// #include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/loop.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
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

// #define __VERFIY_FORWARD_PREDIT__
// #define __VERFIY_INVERSE_DESIGN__
#define __VERIFY_ROUNDTRIP__

#define __Add_PENALTY__

int main(int argc, char* argv[])
{


    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    Config config("cfg.json");
    ActiveComposite ac(config.ResourceSetting.MaterialPath);
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

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


    double scaleFactor_2 = config.RuntimeSetting.Platewidth / (P.colwise().maxCoeff() - P.colwise().minCoeff()).maxCoeff();
    scaleFactor *= scaleFactor_2;
    V *= scaleFactor_2;
    P *= scaleFactor_2;
    geometry.inputVertexPositions *= scaleFactor_2;
    geometry.refreshQuantities();


    std::vector<bool> boundary_vertex_flags(nV, false);
    std::vector<bool> boundary_face_flags(nF, false);
    std::vector<int> boundary_ref_indices(nF, 0);


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
 

    // === Feasibility Diagnostic ===
    {
        spdlog::info("=== Feasibility Diagnostic ===");
        double lam_min = morph_mesh.lambda_pf_t.minCoeff();
        double lam_max = morph_mesh.lambda_pf_t.maxCoeff();
        double lam_mean = morph_mesh.lambda_pf_t.mean();
        double kap_min = morph_mesh.kappa_pv_t.minCoeff();
        double kap_max = morph_mesh.kappa_pv_t.maxCoeff();
        double kap_mean = morph_mesh.kappa_pv_t.mean();
        spdlog::info("Target lambda (per-face): min={:.6f}, max={:.6f}, mean={:.6f}", lam_min, lam_max, lam_mean);
        spdlog::info("Target kappa (per-vertex): min={:.6f}, max={:.6f}, mean={:.6f}", kap_min, kap_max, kap_mean);

        double feas_lam_min = *std::min_element(ac.feasible_lamb.begin(), ac.feasible_lamb.end());
        double feas_lam_max = *std::max_element(ac.feasible_lamb.begin(), ac.feasible_lamb.end());
        double feas_kap_min = *std::min_element(ac.feasible_kapp.begin(), ac.feasible_kapp.end());
        double feas_kap_max = *std::max_element(ac.feasible_kapp.begin(), ac.feasible_kapp.end());
        spdlog::info("Feasible lambda: [{:.6f}, {:.6f}]", feas_lam_min, feas_lam_max);
        spdlog::info("Feasible kappa:  [{:.6f}, {:.6f}]", feas_kap_min, feas_kap_max);

        int lam_in = 0, kap_in = 0;
        for (int i = 0; i < morph_mesh.lambda_pf_t.size(); ++i)
            if (morph_mesh.lambda_pf_t[i] >= feas_lam_min && morph_mesh.lambda_pf_t[i] <= feas_lam_max)
                lam_in++;
        for (int i = 0; i < morph_mesh.kappa_pv_t.size(); ++i)
            if (morph_mesh.kappa_pv_t[i] >= feas_kap_min && morph_mesh.kappa_pv_t[i] <= feas_kap_max)
                kap_in++;
        spdlog::info("Lambda within feasible: {}/{} ({:.1f}%)", lam_in, (int)morph_mesh.lambda_pf_t.size(),
            100.0 * lam_in / morph_mesh.lambda_pf_t.size());
        spdlog::info("Kappa within feasible: {}/{} ({:.1f}%)", kap_in, (int)morph_mesh.kappa_pv_t.size(),
            100.0 * kap_in / morph_mesh.kappa_pv_t.size());

        // Per-element feasibility gap
        double lam_gap_max = 0, kap_gap_max = 0;
        for (int i = 0; i < morph_mesh.lambda_pf_t.size(); ++i) {
            double v = morph_mesh.lambda_pf_t[i];
            double gap = std::max(0.0, std::max(feas_lam_min - v, v - feas_lam_max));
            lam_gap_max = std::max(lam_gap_max, gap);
        }
        for (int i = 0; i < morph_mesh.kappa_pv_t.size(); ++i) {
            double v = morph_mesh.kappa_pv_t[i];
            double gap = std::max(0.0, std::max(feas_kap_min - v, v - feas_kap_max));
            kap_gap_max = std::max(kap_gap_max, gap);
        }
        spdlog::info("Max lambda gap from feasible: {:.6f}", lam_gap_max);
        spdlog::info("Max kappa gap from feasible: {:.6f}", kap_gap_max);
        spdlog::info("=== End Feasibility Diagnostic ===");
    }

    auto V_targ= V;
    V_targ *= 1.0 / scaleFactor;
    std::string output_mesh_targ_path = config.OutputSetting.OutputPath +
        config.ModelSetting.ModelName + "_targ" + ".obj";
    igl::writeOBJ(output_mesh_targ_path, V_targ, F);


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

#ifdef __VERIFY_ROUNDTRIP__
    spdlog::info("=== Round-Trip Verification Test v2 (Spatially Varying) ===");
    {
        // ================================================================
        // Generate spatially varying feasible material distribution
        // ================================================================
        // For each face, assign (t1, t2) based on face centroid position
        // This creates a smooth, spatially varying, but fully FEASIBLE distribution

        geometry.requireVertexIndices();
        geometry.requireFaceIndices();

        // Compute face centroids in parameterization domain (flat)
        Eigen::MatrixXd face_centroids(nF, 2);
        for (int fi = 0; fi < (int)nF; ++fi) {
            face_centroids.row(fi) = (P.row(F(fi, 0)) + P.row(F(fi, 1)) + P.row(F(fi, 2))) / 3.0;
        }
        // Normalize to [0, 1]
        Eigen::Vector2d cmin = face_centroids.colwise().minCoeff();
        Eigen::Vector2d cmax = face_centroids.colwise().maxCoeff();
        for (int fi = 0; fi < (int)nF; ++fi) {
            face_centroids(fi, 0) = (face_centroids(fi, 0) - cmin(0)) / (cmax(0) - cmin(0));
            face_centroids(fi, 1) = (face_centroids(fi, 1) - cmin(1)) / (cmax(1) - cmin(1));
        }

        // Assign t1 based on x-coordinate, t2 based on y-coordinate
        // Map [0,1] -> discrete t values [0, 1/(count-1), ..., 1]
        FaceData<double> t1_true_pf(mesh);
        FaceData<double> t2_true_pf(mesh);
        FaceData<double> lambda_true_pf(mesh);
        FaceData<double> kappa_true_pf(mesh);  // Per-face kappa (no vertex averaging!)

        {
            int fi = 0;
            for (Face f : mesh.faces()) {
                double nx = face_centroids(fi, 0); // [0,1]
                double ny = face_centroids(fi, 1);

                int i1 = static_cast<int>(std::round(nx * (ac.count - 1)));
                int i2 = static_cast<int>(std::round((1.0 - ny) * (ac.count - 1)));
                i1 = std::max(0, std::min(ac.count - 1, i1));
                i2 = std::max(0, std::min(ac.count - 1, i2));

                double t1 = static_cast<double>(i1) / (ac.count - 1);
                double t2 = static_cast<double>(i2) / (ac.count - 1);

                t1_true_pf[f] = t1;
                t2_true_pf[f] = t2;
                lambda_true_pf[f] = compute_lamb_d(ac.m_strain_curve, t1, t2);
                kappa_true_pf[f] = compute_curv_d(ac.m_strain_curve, ac.thickness, t1, t2);
                fi++;
            }
        }

        // Print distribution statistics
        {
            Eigen::VectorXd lam_vec = lambda_true_pf.toVector();
            Eigen::VectorXd kap_vec = kappa_true_pf.toVector();
            spdlog::info("Generated feasible distribution:");
            spdlog::info("  Lambda (per-face): min={:.6f}, max={:.6f}, mean={:.6f}",
                lam_vec.minCoeff(), lam_vec.maxCoeff(), lam_vec.mean());
            spdlog::info("  Kappa (per-face): min={:.6f}, max={:.6f}, mean={:.6f}",
                kap_vec.minCoeff(), kap_vec.maxCoeff(), kap_vec.mean());
        }

        // ================================================================
        // Forward simulate from flat plate
        // ================================================================
        spdlog::info("Forward simulation with spatially varying feasible distribution...");
        auto simFunc_true = simulationFunction(geometry, MrInv,
            lambda_true_pf, kappa_true_pf,  // Both FaceData!
            E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);

        Eigen::MatrixXd V_flat(nV, 3);
        for (size_t i = 0; i < nV; ++i) {
            V_flat(i, 0) = P(i, 0);
            V_flat(i, 1) = P(i, 1);
            V_flat(i, 2) = 0.0;
        }

        Eigen::MatrixXd Vr_true = V_flat;
        newton(geometry, Vr_true, simFunc_true, 200, config.RuntimeSetting.epsilon, true, fixedIdx);

        double fwd_energy = simFunc_true.eval(
            simFunc_true.x_from_data([&](Vertex v) { return Vr_true.row(geometry.vertexIndices[v]); }));
        spdlog::info("Forward simulation done. Final energy: {:.10f}", fwd_energy);

        // Height range of deformed shape
        spdlog::info("Deformed shape z-range: [{:.4f}, {:.4f}]",
            Vr_true.col(2).minCoeff(), Vr_true.col(2).maxCoeff());

        {
            auto V_out = Vr_true * (1.0 / scaleFactor);
            std::string path = config.OutputSetting.OutputPath + config.ModelSetting.ModelName + "_rt_fwd.obj";
            igl::writeOBJ(path, V_out, F);
            spdlog::info("Saved: {}", path);
        }

        Eigen::MatrixXd targetV_rt = Vr_true;

        // ================================================================
        // TEST 1: No penalty, alternating SGN, start from perturbed params
        // ================================================================
        spdlog::info("========================================");
        spdlog::info("TEST 1: No penalty, recover from perturbed start");
        spdlog::info("========================================");
        {
            // Perturb: lambda +3%, kappa scaled to 50%
            FaceData<double> lambda_test(mesh);
            FaceData<double> kappa_test(mesh);
            for (Face f : mesh.faces()) lambda_test[f] = lambda_true_pf[f] * 1.03;
            for (Face f : mesh.faces()) kappa_test[f] = kappa_true_pf[f] * 0.5;

            Eigen::MatrixXd Vr_test = Vr_true;
            double wM_k = config.RuntimeSetting.wM_kap;
            double wL_k = config.RuntimeSetting.wL_kap;
            double wM_l = config.RuntimeSetting.wM_lam;
            double wL_l = config.RuntimeSetting.wL_lam;

            for (int stage = 0; stage < 5; ++stage) {
                spdlog::info("[T1] Stage {} OptKap (wM={:.4f}, wL={:.4f})", stage, wM_k, wL_k);

                auto adj_kap = adjointFunction_FixLam_OptKapPF(geometry, F, MrInv,
                    lambda_test, E, nu, ac.thickness,
                    config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
                Vr_test = sparse_gauss_newton_FixLam_OptKap(geometry, targetV_rt, Vr_test, MrInv,
                    lambda_test, kappa_test, adj_kap, fixedIdx,
                    config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                    wM_k, wL_k, E, nu, ac.thickness,
                    config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);

                double dist_k = (Vr_test - targetV_rt).squaredNorm() / nV;
                Eigen::VectorXd kap_err = kappa_test.toVector() - kappa_true_pf.toVector();
                spdlog::info("[T1]   kap L2err={:.6f}, dist={:.10f}", kap_err.norm() / std::sqrt((double)nF), dist_k);

                spdlog::info("[T1] Stage {} OptLam (wM={:.4f}, wL={:.4f})", stage, wM_l, wL_l);

                auto adj_lam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv,
                    kappa_test, E, nu, ac.thickness,
                    config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
                Vr_test = sparse_gauss_newton_FixKap_OptLam(geometry, targetV_rt, Vr_test, MrInv,
                    lambda_test, kappa_test, adj_lam, fixedIdx,
                    config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                    wM_l, wL_l, E, nu, ac.thickness,
                    config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);

                double dist_l = (Vr_test - targetV_rt).squaredNorm() / nV;
                Eigen::VectorXd lam_err = lambda_test.toVector() - lambda_true_pf.toVector();
                spdlog::info("[T1]   lam L2err={:.6f}, dist={:.10f}", lam_err.norm() / std::sqrt((double)nF), dist_l);

                wM_k *= 0.5; wL_k *= 0.5;
                wM_l *= 0.5; wL_l *= 0.5;
            }

            Eigen::VectorXd lam_final = lambda_test.toVector();
            Eigen::VectorXd kap_final = kappa_test.toVector();
            Eigen::VectorXd lam_true_vec = lambda_true_pf.toVector();
            Eigen::VectorXd kap_true_vec = kappa_true_pf.toVector();
            double lam_err = (lam_final - lam_true_vec).norm() / lam_true_vec.norm();
            double kap_err = (kap_final - kap_true_vec).norm() / kap_true_vec.norm();
            double dist_final = (Vr_test - targetV_rt).squaredNorm() / nV;
            spdlog::info("[T1] FINAL: lam relErr={:.4f}%, kap relErr={:.4f}%, dist={:.10f}",
                lam_err * 100, kap_err * 100, dist_final);

            {
                auto V_out = Vr_test * (1.0 / scaleFactor);
                std::string path = config.OutputSetting.OutputPath + config.ModelSetting.ModelName + "_rt_T1.obj";
                igl::writeOBJ(path, V_out, F);
            }
        }

        // ================================================================
        // TEST 2: With penalty, alternating SGN, start from perturbed params
        // ================================================================
        spdlog::info("========================================");
        spdlog::info("TEST 2: With penalty, recover from perturbed start");
        spdlog::info("========================================");
        {
            double wP_k = config.RuntimeSetting.wP_kap;
            double wP_l = config.RuntimeSetting.wP_lam;
            double betaP_rt = config.RuntimeSetting.betaP;
            auto penalty_kap_fn = MaterialPenaltyFunctionPerF(geometry, ac.feasible_kapp, betaP_rt);
            auto penalty_lam_fn = MaterialPenaltyFunctionPerF(geometry, ac.feasible_lamb, betaP_rt);

            FaceData<double> lambda_test(mesh);
            FaceData<double> kappa_test(mesh);
            for (Face f : mesh.faces()) lambda_test[f] = lambda_true_pf[f] * 1.03;
            for (Face f : mesh.faces()) kappa_test[f] = kappa_true_pf[f] * 0.5;

            Eigen::MatrixXd Vr_test = Vr_true;
            double wM_k = config.RuntimeSetting.wM_kap;
            double wL_k = config.RuntimeSetting.wL_kap;
            double wM_l = config.RuntimeSetting.wM_lam;
            double wL_l = config.RuntimeSetting.wL_lam;

            for (int stage = 0; stage < 5; ++stage) {
                spdlog::info("[T2] Stage {} OptKap (wP={:.4f})", stage, wP_k);

                auto adj_kap = adjointFunction_FixLam_OptKapPF(geometry, F, MrInv,
                    lambda_test, E, nu, ac.thickness,
                    config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
                Vr_test = sparse_gauss_newton_FixLam_OptKap_Penalty(geometry, targetV_rt, Vr_test, MrInv,
                    lambda_test, kappa_test, adj_kap, penalty_kap_fn, fixedIdx,
                    config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                    wM_k, wL_k, wP_k, E, nu, ac.thickness,
                    config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);

                double dist_k = (Vr_test - targetV_rt).squaredNorm() / nV;
                double pen_k = compute_candidate_diff(ac.feasible_kapp, kappa_test.toVector(), true);
                Eigen::VectorXd kap_err_vec = kappa_test.toVector() - kappa_true_pf.toVector();
                spdlog::info("[T2]   kap L2err={:.6f}, pen_kap={:.6f}, dist={:.10f}",
                    kap_err_vec.norm() / std::sqrt((double)nF), pen_k, dist_k);

                spdlog::info("[T2] Stage {} OptLam (wP={:.4f})", stage, wP_l);

                auto adj_lam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv,
                    kappa_test, E, nu, ac.thickness,
                    config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
                Vr_test = sparse_gauss_newton_FixKap_OptLam_Penalty(geometry, targetV_rt, Vr_test, MrInv,
                    lambda_test, kappa_test, adj_lam, penalty_lam_fn, fixedIdx,
                    config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                    wM_l, wL_l, wP_l, E, nu, ac.thickness,
                    config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);

                double dist_l = (Vr_test - targetV_rt).squaredNorm() / nV;
                double pen_l = compute_candidate_diff(ac.feasible_lamb, lambda_test.toVector(), true);
                Eigen::VectorXd lam_err_vec = lambda_test.toVector() - lambda_true_pf.toVector();
                spdlog::info("[T2]   lam L2err={:.6f}, pen_lam={:.6f}, dist={:.10f}",
                    lam_err_vec.norm() / std::sqrt((double)nF), pen_l, dist_l);

                // BUG-10 fixed weight strategy
                if (pen_k >= config.RuntimeSetting.penalty_threshold) wP_k *= 2.0;
                if (pen_l >= config.RuntimeSetting.penalty_threshold) wP_l *= 2.0;
                wM_k = std::max(wM_k * 0.5, config.RuntimeSetting.wM_kap * 1e-3);
                wL_k = std::max(wL_k * 0.5, config.RuntimeSetting.wL_kap * 1e-3);
                wM_l = std::max(wM_l * 0.5, config.RuntimeSetting.wM_lam * 1e-3);
                wL_l = std::max(wL_l * 0.5, config.RuntimeSetting.wL_lam * 1e-3);

                if (pen_k < config.RuntimeSetting.penalty_threshold && pen_l < config.RuntimeSetting.penalty_threshold)
                    break;
            }

            Eigen::VectorXd lam_final = lambda_test.toVector();
            Eigen::VectorXd kap_final = kappa_test.toVector();
            Eigen::VectorXd lam_true_vec = lambda_true_pf.toVector();
            Eigen::VectorXd kap_true_vec = kappa_true_pf.toVector();
            double lam_err = (lam_final - lam_true_vec).norm() / lam_true_vec.norm();
            double kap_err = (kap_final - kap_true_vec).norm() / kap_true_vec.norm();
            double dist_final = (Vr_test - targetV_rt).squaredNorm() / nV;
            double pen_k_final = compute_candidate_diff(ac.feasible_kapp, kap_final, true);
            double pen_l_final = compute_candidate_diff(ac.feasible_lamb, lam_final, true);
            spdlog::info("[T2] FINAL: lam relErr={:.4f}%, kap relErr={:.4f}%, dist={:.10f}",
                lam_err * 100, kap_err * 100, dist_final);
            spdlog::info("[T2] FINAL: pen_kap={:.6f}, pen_lam={:.6f}", pen_k_final, pen_l_final);

            {
                auto V_out = Vr_test * (1.0 / scaleFactor);
                std::string path = config.OutputSetting.OutputPath + config.ModelSetting.ModelName + "_rt_T2.obj";
                igl::writeOBJ(path, V_out, F);
            }
        }

        // ================================================================
        // TEST 3: No penalty, start from RANDOM initial params (far perturbation)
        // ================================================================
        spdlog::info("========================================");
        spdlog::info("TEST 3: No penalty, far perturbation (mean of feasible set)");
        spdlog::info("========================================");
        {
            // Start from the mean of the feasible set (completely different from truth)
            double lam_mean = 0.0, kap_mean = 0.0;
            for (size_t i = 0; i < ac.feasible_lamb.size(); ++i) {
                lam_mean += ac.feasible_lamb[i];
                kap_mean += ac.feasible_kapp[i];
            }
            lam_mean /= ac.feasible_lamb.size();
            kap_mean /= ac.feasible_kapp.size();

            FaceData<double> lambda_test(mesh, lam_mean);
            FaceData<double> kappa_test(mesh, kap_mean);

            spdlog::info("Initial: uniform lambda={:.6f}, kappa={:.6f} (mean of feasible set)", lam_mean, kap_mean);

            Eigen::MatrixXd Vr_test = V_flat; // start from flat plate
            double wM_k = config.RuntimeSetting.wM_kap;
            double wL_k = config.RuntimeSetting.wL_kap;
            double wM_l = config.RuntimeSetting.wM_lam;
            double wL_l = config.RuntimeSetting.wL_lam;

            for (int stage = 0; stage < 5; ++stage) {
                spdlog::info("[T3] Stage {} OptKap (wM={:.4f}, wL={:.4f})", stage, wM_k, wL_k);

                auto adj_kap = adjointFunction_FixLam_OptKapPF(geometry, F, MrInv,
                    lambda_test, E, nu, ac.thickness,
                    config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
                Vr_test = sparse_gauss_newton_FixLam_OptKap(geometry, targetV_rt, Vr_test, MrInv,
                    lambda_test, kappa_test, adj_kap, fixedIdx,
                    config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                    wM_k, wL_k, E, nu, ac.thickness,
                    config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);

                double dist_k = (Vr_test - targetV_rt).squaredNorm() / nV;
                Eigen::VectorXd kap_err_vec = kappa_test.toVector() - kappa_true_pf.toVector();
                spdlog::info("[T3]   kap L2err={:.6f}, dist={:.10f}",
                    kap_err_vec.norm() / std::sqrt((double)nF), dist_k);

                spdlog::info("[T3] Stage {} OptLam (wM={:.4f}, wL={:.4f})", stage, wM_l, wL_l);

                auto adj_lam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv,
                    kappa_test, E, nu, ac.thickness,
                    config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
                Vr_test = sparse_gauss_newton_FixKap_OptLam(geometry, targetV_rt, Vr_test, MrInv,
                    lambda_test, kappa_test, adj_lam, fixedIdx,
                    config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                    wM_l, wL_l, E, nu, ac.thickness,
                    config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);

                double dist_l = (Vr_test - targetV_rt).squaredNorm() / nV;
                Eigen::VectorXd lam_err_vec = lambda_test.toVector() - lambda_true_pf.toVector();
                spdlog::info("[T3]   lam L2err={:.6f}, dist={:.10f}",
                    lam_err_vec.norm() / std::sqrt((double)nF), dist_l);

                wM_k *= 0.5; wL_k *= 0.5;
                wM_l *= 0.5; wL_l *= 0.5;
            }

            Eigen::VectorXd lam_final = lambda_test.toVector();
            Eigen::VectorXd kap_final = kappa_test.toVector();
            Eigen::VectorXd lam_true_vec = lambda_true_pf.toVector();
            Eigen::VectorXd kap_true_vec = kappa_true_pf.toVector();
            double lam_err = (lam_final - lam_true_vec).norm() / lam_true_vec.norm();
            double kap_err = (kap_final - kap_true_vec).norm() / kap_true_vec.norm();
            double dist_final = (Vr_test - targetV_rt).squaredNorm() / nV;
            spdlog::info("[T3] FINAL: lam relErr={:.4f}%, kap relErr={:.4f}%, dist={:.10f}",
                lam_err * 100, kap_err * 100, dist_final);

            {
                auto V_out = Vr_test * (1.0 / scaleFactor);
                std::string path = config.OutputSetting.OutputPath + config.ModelSetting.ModelName + "_rt_T3.obj";
                igl::writeOBJ(path, V_out, F);
            }
        }

        spdlog::info("=== End Round-Trip Verification v2 ===");
    }

#else // !__VERIFY_ROUNDTRIP__

    spdlog::info("Step 4: Inverse Design.");

    double wP_kap = config.RuntimeSetting.wP_kap;
    double wP_lam = config.RuntimeSetting.wP_lam;
    double penalty_threshold = config.RuntimeSetting.penalty_threshold;
    double betaP = config.RuntimeSetting.betaP;
    auto penalty_to_lamb = MaterialPenaltyFunctionPerF(geometry, ac.feasible_lamb, betaP);
    auto penalty_to_kapp = MaterialPenaltyFunctionPerF(geometry, ac.feasible_kapp, betaP);
    auto penalty_to_modu = MaterialPenaltyFunctionPerV(geometry, ac.feasible_modl, betaP);

    int stage_iter = 5;
    int k = 0;

    double wM_kap = config.RuntimeSetting.wM_kap;
    double wM_lam = config.RuntimeSetting.wM_lam;
    double wL_kap = config.RuntimeSetting.wL_kap;
    double wL_lam = config.RuntimeSetting.wL_lam;

#ifdef __Add_PENALTY__

    double distance = 0.0;
    double penalty_kap = 0.0;
    double penalty_lam = 0.0;
    while(k < stage_iter)
    {
        spdlog::info("Stage {}, OptKap start, wP_kap: {:.6f}, wP_lam: {:.6f}.", k, wP_kap, wP_lam);
        // Vr = targetV;
        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKapPF(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
        Vr = sparse_gauss_newton_FixLam_OptKap_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, adjointFunc_OptKap, penalty_to_kapp, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_kap, wL_kap, wP_kap,
            E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b);

        // Compute and output distance and penalties after OptKap
        distance = (Vr - targetV).squaredNorm() / nV;
        penalty_kap = compute_candidate_diff(ac.feasible_kapp,kappa_pf_s.toVector(),true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb,lambda_pf_s.toVector(),true);
        spdlog::info("Stage {}, OptKap finish - Distance: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                     k, distance, penalty_kap, penalty_lam);



        spdlog::info("Stage {}, OptLam start, wP_kap: {:.6f}, wP_lam: {:.6f}.", k, wP_kap, wP_lam);
        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
        Vr = sparse_gauss_newton_FixKap_OptLam_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, adjointFunc_OptLam, penalty_to_lamb, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, wM_lam, wL_lam, wP_lam,
            E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b);

        // Compute and output distance and penalties after OptLam
        distance = (Vr - targetV).squaredNorm() / nV;
        penalty_kap = compute_candidate_diff(ac.feasible_kapp,kappa_pf_s.toVector(),true);
        penalty_lam = compute_candidate_diff(ac.feasible_lamb,lambda_pf_s.toVector(),true);
        spdlog::info("Stage {}, OptLam finish- Distance: {:.6f}, Penalty_kap: {:.6f}, Penalty_lam: {:.6f}",
                     k, distance, penalty_kap, penalty_lam);

        // Evaluate distance after jointly projecting kappa and lambda to the same feasible index
        {
            FaceData<double> kappa_pf_proj(mesh);
            FaceData<double> lambda_pf_proj(mesh);

            for (Face f : mesh.faces()) {
                double kap = kappa_pf_s[f];
                double lam = lambda_pf_s[f];
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap, lam);
                kappa_pf_proj[f] = ac.feasible_kapp[idx];
                lambda_pf_proj[f] = ac.feasible_lamb[idx];
            }

            auto simFunc_proj = simulationFunction(geometry, MrInv, lambda_pf_proj, kappa_pf_proj,
                E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
            Eigen::MatrixXd Vr_proj = Vr;
            newton(geometry, Vr_proj, simFunc_proj,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx);
            double dist_proj = (Vr_proj - targetV).squaredNorm() / nV;

            spdlog::info("Stage {}, Projected distance: {:.6f}", k, dist_proj);
        }

        k++;

        // BUG-10 fix: adaptive weight strategy (gentler than original 10x/0.1x)
        if(penalty_kap < penalty_threshold && penalty_lam < penalty_threshold)
            break;

        // Penalty growth: gentle 2x (was 10x)
        if (penalty_kap >= penalty_threshold) {
            wP_kap *= 2.0;
        }
        if (penalty_lam >= penalty_threshold) {
            wP_lam *= 2.0;
        }

        // Regularization decay: gentle 0.5x (was 0.1x) with floor
        double wM_kap_floor = config.RuntimeSetting.wM_kap * 1e-3;
        double wL_kap_floor = config.RuntimeSetting.wL_kap * 1e-3;
        double wM_lam_floor = config.RuntimeSetting.wM_lam * 1e-3;
        double wL_lam_floor = config.RuntimeSetting.wL_lam * 1e-3;
        wM_kap = std::max(wM_kap * 0.5, wM_kap_floor);
        wL_kap = std::max(wL_kap * 0.5, wL_kap_floor);
        wM_lam = std::max(wM_lam * 0.5, wM_lam_floor);
        wL_lam = std::max(wL_lam * 0.5, wL_lam_floor);
    }


#else

    Vr = targetV;
    while(k < stage_iter)
    {
        spdlog::info("Stage {}, OptKap start", k);
        
        auto adjointFunc_OptKap = adjointFunction_FixLam_OptKapPF(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
        Vr = sparse_gauss_newton_FixLam_OptKap(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, adjointFunc_OptKap, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, config.RuntimeSetting.wM, config.RuntimeSetting.wL,
            E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b);

        // Compute and output distance after OptKap
        double distance_kap = (Vr - targetV).squaredNorm() / nV;
        spdlog::info("Stage {}, OptKap finish - Distance: {:.6f}", k, distance_kap);


        spdlog::info("Stage {}, OptLam start", k);
        auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b);
        Vr = sparse_gauss_newton_FixKap_OptLam(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, adjointFunc_OptLam, fixedIdx,
            config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, 0.0, config.RuntimeSetting.wL,
            E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b);

        // morph_mesh.lambda_pv_s = lambda_pv_s.toVector();
        // morph_mesh.kappa_pv_s = kappa_pv_s.toVector();

        // // Reassign per-face values from optimized per-vertex values
        // Morphmesh::ReassignPFFromPV(geometry, F, morph_mesh.lambda_pv_s, morph_mesh.kappa_pv_s,
        //     morph_mesh.lambda_pf_s, morph_mesh.kappa_pf_s);
        // lambda_pf_s = FaceData<double>(mesh, morph_mesh.lambda_pf_s);
        // kappa_pf_s = FaceData<double>(mesh, morph_mesh.kappa_pf_s);

        // Compute and output distance after OptLam
        double distance_lam = (Vr - targetV).squaredNorm() / nV;
        spdlog::info("Stage {}, OptLam finish - Distance: {:.6f}", k, distance_lam);


        k++;
    }

    
#endif // __Add_PENALTY__

    auto V_inv = Vr;
    V_inv *= 1.0 / scaleFactor;
    std::string output_mesh_inv_path = config.OutputSetting.OutputPath +
        config.ModelSetting.ModelName + "_inv" + ".obj";
    igl::writeOBJ(output_mesh_inv_path, V_inv, F);

#endif // __VERIFY_ROUNDTRIP__

    ///***************************************** View by Imgui *****************************************///               
    //igl::opengl::glfw::Viewer viewer;
    //viewer.data().set_mesh(V_pred, F);
    ////viewer.data().set_colors(C);
    //viewer.data().show_lines = true;
    //viewer.launch();

    spdlog::info("program finish.");

}

