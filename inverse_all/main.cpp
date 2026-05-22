

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
// #include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/loop.h>
#include <igl/cotmatrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <numeric>
#include <filesystem>
#include <io.h>

#include <spdlog/spdlog.h>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include"config.hpp"
#include "material.hpp"
#include "parameterization.h"
#include "simulation_utils.h"
#include "functions.h"
#include "newton.h"
#include "LocalGlobalSolver.h"
#include "morphmesh.hpp"
#include "morph_functions.hpp"
#include "output.hpp"
#include "boundary_utils.h"

// #define __VERFIY_FORWARD_PREDIT__
#define __VERFIY_INVERSE_DESIGN__

#define __Add_PENALTY__

int main(int argc, char* argv[])
{


    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    Config config("cfg.json");
    ActiveComposite ac(config.materialJsonPath());
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    spdlog::info("program start (multi-patch inverse design):");

    ///***************************************** Patch I/O setup *****************************************///

    const std::string model       = config.ModelSetting.ModelName;
    const std::string target_dir  = config.PathSetting.TargetDir + model + "/";
    const std::string param_dir   = config.PathSetting.ParamDir  + model + "/";
    const std::string cond_dir    = config.PathSetting.CondDir   + model + "/";
    const std::string design_dir  = config.PathSetting.DesignDir + model + "/";
    const std::string morph_dir   = config.PathSetting.MorphDir  + model + "/";
    std::filesystem::create_directories(morph_dir);
    std::filesystem::create_directories(design_dir);
    spdlog::info("Target dir : {}", target_dir);
    spdlog::info("Param  dir : {}", param_dir);
    spdlog::info("Cond   dir : {}", cond_dir);
    spdlog::info("Design dir : {}", design_dir);
    spdlog::info("Morph  dir : {}", morph_dir);

    ///***************************************** Phase 1: read pre-scaled V/P from disk *****************************************///
    // V comes from TargetDir (scaled), P comes from ParamDir (scaled).  ParamAll
    // must have produced these + global_scale.txt + per-patch cond files.

    struct PatchData {
        size_t idx;
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        Eigen::MatrixXd P;
        size_t nV, nF;
    };
    std::vector<PatchData> patches;

    for (size_t i = 0;; ++i) {
        std::string v_path = target_dir + "patch_" + std::to_string(i) + "_V.obj";
        std::string p_path = param_dir  + "patch_" + std::to_string(i) + "_P.obj";
        Eigen::MatrixXd Vp;
        Eigen::MatrixXi Fp;
        if (!igl::readOBJ(v_path, Vp, Fp)) {
            spdlog::info("No more patches after idx {}.", i - 1);
            break;
        }
        Eigen::MatrixXd P_loaded3;
        Eigen::MatrixXi F_p;
        if (!igl::readOBJ(p_path, P_loaded3, F_p)) {
            spdlog::error("Patch {} V exists but P missing: {}.  Run ParamAll first.", i, p_path);
            return -1;
        }
        if (P_loaded3.rows() != Vp.rows()) {
            spdlog::error("Patch {}: P vertex count ({}) != V count ({}).",
                          i, P_loaded3.rows(), Vp.rows());
            return -1;
        }

        PatchData pd;
        pd.idx = i;
        pd.V = std::move(Vp);
        pd.F = std::move(Fp);
        pd.P = P_loaded3.leftCols(2);
        pd.nV = pd.V.rows();
        pd.nF = pd.F.rows();
        spdlog::info("Patch {}: read {} V, {} F.", i, pd.nV, pd.nF);
        patches.push_back(std::move(pd));
    }
    spdlog::info("Total patches: {}", patches.size());

    if (patches.empty()) {
        spdlog::error("No patches found, abort.");
        return -1;
    }

    ///***************************************** Phase 2: per-patch inverse design *****************************************///
    // V/P are already in physical (device) units; no re-scaling here.

    for (auto& pd : patches) {
        spdlog::info("=================================================");
        spdlog::info("Inverse design for patch {} ({} V, {} F)", pd.idx, pd.nV, pd.nF);
        spdlog::info("=================================================");

        // V and P are already scaled by globalScale (done by ParamAll).
        // Aliases so the existing inverse-design code below reads naturally.
        Eigen::MatrixXd& V = pd.V;
        Eigen::MatrixXi& F = pd.F;
        Eigen::MatrixXd& P = pd.P;
        size_t nV = pd.nV;
        size_t nF = pd.nF;

        ManifoldSurfaceMesh mesh(F);
        VertexPositionGeometry geometry(mesh, V);
        geometry.refreshQuantities();

        // Boundary-face reference mapping (BFS on dual graph).
        std::vector<bool> is_boundary_face;
        std::vector<int> ref_faces = buildRefFaces(mesh, is_boundary_face);

        ///***************************************** Material Settings *****************************************///

        spdlog::info("Step 2: Material Settings.");

        Eigen::MatrixXd targetV = V;

        FaceData<Eigen::MatrixXd> M = precomputeM(mesh, V, F);
        FaceData<Eigen::Matrix2d> MrInv = precomputeMrInv(mesh, P, F);

        // Boundary condition: read 3 vertex indices from cond file written by ParamAll
        std::vector<int> fixedVertexIdx;
        {
            const std::string cond_path = cond_dir + "patch_" + std::to_string(pd.idx) + "_bound_center.txt";
            std::ifstream ifs(cond_path);
            if (!ifs.is_open()) {
                spdlog::error("Cannot read cond file: {}.  Run ParamAll first.", cond_path);
                return -1;
            }
            int v0, v1, v2;
            ifs >> v0 >> v1 >> v2;
            fixedVertexIdx = {v0, v1, v2};
            spdlog::info("Patch {} cond: v0={} v1={} v2={}", pd.idx, v0, v1, v2);
        }
        std::vector<int> fixedIdx;  // 9 DOF indices
        for (int v : fixedVertexIdx)
            for (int k = 0; k < 3; ++k) fixedIdx.push_back(3 * v + k);
        std::sort(fixedIdx.begin(), fixedIdx.end());

        double E = 1.0;
        double nu = 0.5;
        Morphmesh morph_mesh(V, P, F, E, nu);
        Morphmesh::ComputeMorphophing(geometry, V, F, nV, nF, ref_faces,
            MrInv, morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t, morph_mesh.kappa_pv_t, morph_mesh.kappa_pf_t, &morph_mesh.vertex_area_sum);
        Morphmesh::SetMorphophing(morph_mesh.lambda_pv_t, morph_mesh.lambda_pf_t,
            morph_mesh.kappa_pv_t,morph_mesh.kappa_pf_t,
            morph_mesh.lambda_pv_s, morph_mesh.lambda_pf_s,
            morph_mesh.kappa_pf_s, morph_mesh.kappa_pf_s);

        // Mirror physical-unit target into morph_dir so {targ, inv} sit side-by-side.
        igl::writeOBJ(morph_dir + "patch_" + std::to_string(pd.idx) + "_targ.obj", V, F);


        VertexData<double> lambda_pv_s(mesh, morph_mesh.lambda_pv_s);
        FaceData<double> lambda_pf_s(mesh, morph_mesh.lambda_pf_s);
        FaceData<double> kappa_pf_s(mesh, morph_mesh.kappa_pf_s);

        ///***************************************** Inverse Design *****************************************///

        auto V_pred = V;
        // V_init: flat plate aligned so fixed vertices match V exactly.
        Eigen::MatrixXd Vr = flatPlateAligned(P, V, fixedVertexIdx);
        igl::writeOBJ(morph_dir + "patch_" + std::to_string(pd.idx) + "_init.obj", Vr, F);

        // Lumped vertex mass vector (size 3*nV) -- shared by all SGN calls so
        // distance/SPN values match between main and the solver exactly.
        const Eigen::VectorXd masses = computeVertexMasses(geometry);

        // Face-space matrices to compute the other-variable regulariser as
        // a constant offset, so OptKap/OptLam stages share a unified SPN
        // energy formula (distance + kappa_reg + lambda_reg).
        // M_kappa depends on MrInv -> refreshed on every P-update.
        Eigen::SparseMatrix<double> M_kappa = computeFaceMassKappa(mesh, MrInv);
        const Eigen::SparseMatrix<double> M_lambda = computeFaceMassLambda(geometry);
        const Eigen::SparseMatrix<double> L_face = computeFaceDualLaplacian(mesh);

        LocalGlobalSolver paramSolver(V, F);   // legacy; unused by SGN OptP

        // P-side regularisation matrices (size 2|V| x 2|V|), see inverse/main.
        Eigen::SparseMatrix<double> M_P_2(2 * nV, 2 * nV);
        M_P_2.setIdentity();
        Eigen::SparseMatrix<double> L_P_2(2 * nV, 2 * nV);
        {
            Eigen::SparseMatrix<double> L_v;
            igl::cotmatrix(V, F, L_v);
            L_v = (-L_v).eval();
            std::vector<Eigen::Triplet<double>> trips;
            trips.reserve(L_v.nonZeros() * 2);
            for (int i = 0; i < L_v.outerSize(); ++i)
                for (Eigen::SparseMatrix<double>::InnerIterator it(L_v, i); it; ++it)
                    for (int d = 0; d < 2; ++d)
                        trips.emplace_back(2 * (int)it.row() + d, 2 * (int)it.col() + d, it.value());
            L_P_2.setFromTriplets(trips.begin(), trips.end());
        }
        const Eigen::MatrixXd P_anchor = P;

        spdlog::info("Step 4: Inverse Design.");

        double wP_lam = config.RuntimeSetting.wP_lam;
        double wP_kap = config.RuntimeSetting.wP_kap;
        double penalty_threshold = config.RuntimeSetting.penalty_threshold;
        double betaP = config.RuntimeSetting.betaP;
        // 1D independent hard-min penalties (per-direction wP).
        auto penalty_to_kapp = MaterialPenaltyFunctionPerF(geometry, ac.feasible_kapp, betaP);
        auto penalty_to_lamb = MaterialPenaltyFunctionPerF(geometry, ac.feasible_lamb, betaP);
        auto penalty_to_modu = MaterialPenaltyFunctionPerV(geometry, ac.feasible_modl, betaP);

        int stage_iter = config.RuntimeSetting.stage_iter;
        int k = 0;

        double wM_kap = config.RuntimeSetting.wM_kap;
        double wM_lam = config.RuntimeSetting.wM_lam;
        double wL_kap = config.RuntimeSetting.wL_kap;
        double wL_lam = config.RuntimeSetting.wL_lam;

#ifdef __Add_PENALTY__

        double distance = 0.0;
        double spn_energy = 0.0;
        double penalty_kap = 0.0;
        double penalty_lam = 0.0;

        // Anchor for mass reg.  kappa_anchor = 0 (kappa natural rest = 0).
        // lambda_anchor = mean of feasible candidates (grayscale lam > 1, so
        // pulling toward 0 is unphysical).  L * 1 = 0 means the Laplacian
        // term is anchor-invariant, only the mass term needs the shift.
        const double kappa_anchor  = 0.0;
        const double lambda_anchor = std::accumulate(ac.feasible_lamb.begin(),
                                                      ac.feasible_lamb.end(), 0.0)
                                     / static_cast<double>(ac.feasible_lamb.size());

        auto computeKappaReg = [&]() {
            const Eigen::VectorXd k  = kappa_pf_s.toVector();
            const Eigen::VectorXd ko = k - Eigen::VectorXd::Constant(k.size(), kappa_anchor);
            return wM_kap * ko.dot(M_kappa * ko) + wL_kap * k.dot(L_face * k);
        };
        auto computeLambdaReg = [&]() {
            const Eigen::VectorXd l  = lambda_pf_s.toVector();
            const Eigen::VectorXd lo = l - Eigen::VectorXd::Constant(l.size(), lambda_anchor);
            return wM_lam * lo.dot(M_lambda * lo) + wL_lam * l.dot(L_face * l);
        };
        double kappa_reg  = computeKappaReg();
        double lambda_reg = computeLambdaReg();
        double self_reg   = 0.0;

        // Snap (lambda, kappa) per face, run forward Newton, return both
        // mass-weighted distance and the resulting Vr_proj.  bd/int RMS
        // callers reuse V_proj for free.
        struct ProjState { double dist; Eigen::MatrixXd V; };
        auto computeProjStateFrom = [&](const Eigen::MatrixXd& Vr_start) -> ProjState {
            FaceData<double> kappa_pf_proj(mesh);
            FaceData<double> lambda_pf_proj(mesh);
            for (Face f : mesh.faces()) {
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb,
                                            kappa_pf_s[f], lambda_pf_s[f]);
                kappa_pf_proj[f]  = ac.feasible_kapp[idx];
                lambda_pf_proj[f] = ac.feasible_lamb[idx];
            }
            // Recompute MrInv from the current P here -- inside OptP, the
            // outer `MrInv` lags behind P during SGN inner iterations.
            FaceData<Eigen::Matrix2d> MrInv_curr = precomputeMrInv(mesh, P, F);
            auto simFunc_proj = simulationFunction(geometry, MrInv_curr, lambda_pf_proj, kappa_pf_proj,
                E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Eigen::MatrixXd Vr_proj = Vr_start;
            newton(geometry, Vr_proj, simFunc_proj,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx);
            double d2 = 0.0;
            for (size_t i = 0; i < nV; ++i)
                for (int j = 0; j < 3; ++j) {
                    double d = Vr_proj(i, j) - targetV(i, j);
                    d2 += masses(3 * i + j) * d * d;
                }
            return { d2, std::move(Vr_proj) };
        };
        auto computeProjectedDistanceFrom = [&](const Eigen::MatrixXd& Vr_start) {
            return computeProjStateFrom(Vr_start).dist;
        };
        auto computeProjectedDistance = [&]() { return computeProjectedDistanceFrom(Vr); };
        // Per-vertex Euclidean RMS, separated by boundary / interior.
        geometry.requireVertexIndices();
        auto computeBoundaryInteriorRMS = [&](const Eigen::MatrixXd& V_cur) -> std::pair<double, double> {
            double bd_sum2 = 0.0, int_sum2 = 0.0;
            int    bd_cnt  = 0,   int_cnt  = 0;
            for (Vertex v : mesh.vertices()) {
                const int vi = static_cast<int>(geometry.vertexIndices[v]);
                const Eigen::Vector3d d = V_cur.row(vi) - targetV.row(vi);
                const double d2 = d.squaredNorm();
                if (v.isBoundary()) { bd_sum2 += d2; ++bd_cnt; }
                else                { int_sum2 += d2; ++int_cnt; }
            }
            const double bd_rms  = bd_cnt  > 0 ? std::sqrt(bd_sum2 / bd_cnt)   : 0.0;
            const double int_rms = int_cnt > 0 ? std::sqrt(int_sum2 / int_cnt) : 0.0;
            return { bd_rms, int_rms };
        };
        auto reshape_x_to_V = [&](const Eigen::VectorXd& x_vec) {
            Eigen::MatrixXd V_iter(nV, 3);
            for (size_t v = 0; v < nV; ++v)
                for (int j = 0; j < 3; ++j)
                    V_iter(v, j) = x_vec(3 * v + j);
            return V_iter;
        };

        // Re-run forward Newton on current (P, lambda, kappa) to bring Vr back
        // to equilibrium after a P-update changed MrInv.
        auto recomputeForwardState = [&]() -> double {
            auto simFunc = simulationFunction(geometry, MrInv, lambda_pf_s, kappa_pf_s,
                E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            newton(geometry, Vr, simFunc,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx);
            double d2 = 0.0;
            for (size_t i = 0; i < nV; ++i)
                for (int j = 0; j < 3; ++j) {
                    double d = Vr(i, j) - targetV(i, j);
                    d2 += masses(3 * i + j) * d * d;
                }
            return d2;
        };

        // ---- Trust-region best snapshot ------------------------------------
        double dist_best = std::numeric_limits<double>::infinity();
        double proj_best = std::numeric_limits<double>::infinity();
        Eigen::MatrixXd            Vr_best         = Vr;
        FaceData<double>           lambda_pf_best  = lambda_pf_s;
        FaceData<double>           kappa_pf_best   = kappa_pf_s;
        Eigen::MatrixXd            P_best          = P;
        FaceData<Eigen::Matrix2d>  MrInv_best      = MrInv;
        Eigen::SparseMatrix<double> M_kappa_best   = M_kappa;
        // wM / wL are homotopy-schedule, not rolled back on REJECT.
        double wP_lam_best = wP_lam, wP_kap_best = wP_kap;
        double kappa_reg_best  = kappa_reg;
        double lambda_reg_best = lambda_reg;
        double wP_lam_growth_factor = config.RuntimeSetting.wP_lam_growth_factor;
        double wP_kap_growth_factor = config.RuntimeSetting.wP_kap_growth_factor;

        // Per-patch CSV log: written under MorphLogsDir/{method}/{model}/.
        // Schema: stage,substage,iter,spn,dist.
        const std::string morphlogs_dir = config.PathSetting.MorphLogsDir
                                        + config.RuntimeSetting.morph_method + "/"
                                        + model + "/";
        std::filesystem::create_directories(morphlogs_dir);
        std::ofstream iter_log_ofs(morphlogs_dir + "patch_" + std::to_string(pd.idx) + "_iter_log.csv");
        iter_log_ofs << "stage,substage,iter,spn,dist,proj_dist,"
                     << "kappa_reg,lambda_reg,penalty_kap,penalty_lam,"
                     << "wP_kap,wP_lam,wM_kap,wL_kap,wM_lam,wL_lam,"
                     << "bd_rms,int_rms\n";
        spdlog::info("Patch {} iter log -> {}", pd.idx,
                     morphlogs_dir + "patch_" + std::to_string(pd.idx) + "_iter_log.csv");

        while (k < stage_iter)
        {
            spdlog::info("Patch {} Stage {}: wP_lam={:.6f}, wP_kap={:.6f}, wM_kap={:.6f}, wL_kap={:.6f}, wM_lam={:.6f}, wL_lam={:.6f}",
                         pd.idx, k, wP_lam, wP_kap, wM_kap, wL_kap, wM_lam, wL_lam);

            // -- OptKap --
            auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            auto logger_OptKap = [&](int i, const Eigen::VectorXd& x_iter,
                                     double spn, double dist, double self_reg_iter, double /*pen*/) {
                const auto _proj_st = computeProjStateFrom(reshape_x_to_V(x_iter));
                const double pd      = _proj_st.dist;
                const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(_proj_st.V);
                const double pen_kap = penalty_to_kapp.eval(kappa_pf_s.toVector());
                const double pen_lam = penalty_to_lamb.eval(lambda_pf_s.toVector());
                std::cout << "\tproj=" << pd << "\tbd=" << bd_rms << "\tint=" << int_rms << std::endl;
                iter_log_ofs << k << ",OptKap," << i << ","
                             << spn << "," << dist << "," << pd << ","
                             << self_reg_iter << "," << lambda_reg << ","
                             << pen_kap << "," << pen_lam << ","
                             << wP_kap << "," << wP_lam << ","
                             << wM_kap << "," << wL_kap << "," << wM_lam << "," << wL_lam << "\n";
            };
            Vr = sparse_gauss_newton_FixLam_OptKap_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, lambda_reg,
                adjointFunc_OptKap, penalty_to_kapp, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                wM_kap, wL_kap, kappa_anchor, wP_kap,
                E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                distance, spn_energy, self_reg,
                logger_OptKap);
            kappa_reg = self_reg;
            {
                const auto _proj_st = computeProjStateFrom(Vr);
                const double pd_end  = _proj_st.dist;
                const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(_proj_st.V);
                const double pen_kap = penalty_to_kapp.eval(kappa_pf_s.toVector());
                const double pen_lam = penalty_to_lamb.eval(lambda_pf_s.toVector());
                iter_log_ofs << k << ",OptKap,-1," << spn_energy << "," << distance << "," << pd_end << ","
                             << kappa_reg << "," << lambda_reg << ","
                             << pen_kap << "," << pen_lam << ","
                             << wP_kap << "," << wP_lam << ","
                             << wM_kap << "," << wL_kap << "," << wM_lam << "," << wL_lam << "\n";
            }
            penalty_kap = compute_candidate_diff(ac.feasible_kapp, kappa_pf_s.toVector(), true);
            penalty_lam = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
            {
                const auto _st = computeProjStateFrom(Vr);
                const auto [_bd, _int] = computeBoundaryInteriorRMS(_st.V);
                spdlog::info("Patch {} Stage {} [OptKap finish] SPN={:.6f} Dist={:.6f} ProjDist={:.6f} bd_rms={:.6f} int_rms={:.6f} Pkap={:.6f} Plam={:.6f}",
                             pd.idx, k, spn_energy, distance, _st.dist, _bd, _int, penalty_kap, penalty_lam);
            }

            // -- OptLam --
            auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            auto logger_OptLam = [&](int i, const Eigen::VectorXd& x_iter,
                                     double spn, double dist, double self_reg_iter, double /*pen*/) {
                const auto _proj_st = computeProjStateFrom(reshape_x_to_V(x_iter));
                const double pd      = _proj_st.dist;
                const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(_proj_st.V);
                const double pen_kap = penalty_to_kapp.eval(kappa_pf_s.toVector());
                const double pen_lam = penalty_to_lamb.eval(lambda_pf_s.toVector());
                std::cout << "\tproj=" << pd << "\tbd=" << bd_rms << "\tint=" << int_rms << std::endl;
                iter_log_ofs << k << ",OptLam," << i << ","
                             << spn << "," << dist << "," << pd << ","
                             << kappa_reg << "," << self_reg_iter << ","
                             << pen_kap << "," << pen_lam << ","
                             << wP_kap << "," << wP_lam << ","
                             << wM_kap << "," << wL_kap << "," << wM_lam << "," << wL_lam << "\n";
            };
            Vr = sparse_gauss_newton_FixKap_OptLam_Penalty(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, kappa_reg,
                adjointFunc_OptLam, penalty_to_lamb, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                wM_lam, wL_lam, lambda_anchor, wP_lam,
                E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                distance, spn_energy, self_reg,
                logger_OptLam);
            lambda_reg = self_reg;
            {
                const auto _proj_st = computeProjStateFrom(Vr);
                const double pd_end  = _proj_st.dist;
                const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(_proj_st.V);
                const double pen_kap = penalty_to_kapp.eval(kappa_pf_s.toVector());
                const double pen_lam = penalty_to_lamb.eval(lambda_pf_s.toVector());
                iter_log_ofs << k << ",OptLam,-1," << spn_energy << "," << distance << "," << pd_end << ","
                             << kappa_reg << "," << lambda_reg << ","
                             << pen_kap << "," << pen_lam << ","
                             << wP_kap << "," << wP_lam << ","
                             << wM_kap << "," << wL_kap << "," << wM_lam << "," << wL_lam << "\n";
            }
            penalty_kap = compute_candidate_diff(ac.feasible_kapp, kappa_pf_s.toVector(), true);
            penalty_lam = compute_candidate_diff(ac.feasible_lamb, lambda_pf_s.toVector(), true);
            {
                const auto _st = computeProjStateFrom(Vr);
                const auto [_bd, _int] = computeBoundaryInteriorRMS(_st.V);
                spdlog::info("Patch {} Stage {} [OptLam finish] SPN={:.6f} Dist={:.6f} ProjDist={:.6f} bd_rms={:.6f} int_rms={:.6f} Pkap={:.6f} Plam={:.6f}",
                             pd.idx, k, spn_energy, distance, _st.dist, _bd, _int, penalty_kap, penalty_lam);
            }

            // -- Optional joint snap (lambda, kappa) before P-update --
            if (config.RuntimeSetting.snap_before_P) {
                for (Face f : mesh.faces()) {
                    int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb,
                                                kappa_pf_s[f], lambda_pf_s[f]);
                    lambda_pf_s[f] = ac.feasible_lamb[idx];
                    kappa_pf_s[f]  = ac.feasible_kapp[idx];
                }
            }

            // -- OptP via SGN (distance-driven) --
            {
                auto adjointFunc_OptP = adjointFunction_FixMaterial_OptP(
                    geometry, F, lambda_pf_s, kappa_pf_s,
                    E, nu, ac.thickness,
                    config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);

                const double wM_P = config.RuntimeSetting.wM_P;
                const double wL_P = config.RuntimeSetting.wL_P;

                auto logger_OptP = [&](int i, const Eigen::VectorXd& x_iter,
                                       double spn, double dist, double /*self_reg*/, double /*pen*/) {
                    const auto _proj_st = computeProjStateFrom(reshape_x_to_V(x_iter));
                const double pd      = _proj_st.dist;
                const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(_proj_st.V);
                    const double pen_kap = penalty_to_kapp.eval(kappa_pf_s.toVector());
                    const double pen_lam = penalty_to_lamb.eval(lambda_pf_s.toVector());
                    std::cout << "\tproj=" << pd << "\tbd=" << bd_rms << "\tint=" << int_rms << std::endl;
                    iter_log_ofs << k << ",OptP," << i << ","
                                 << spn << "," << dist << "," << pd << ","
                                 << kappa_reg << "," << lambda_reg << ","
                                 << pen_kap << "," << pen_lam << ","
                                 << wP_kap << "," << wP_lam << ","
                                 << wM_kap << "," << wL_kap << "," << wM_lam << "," << wL_lam << "\n";
                };

                double P_reg = 0.0;
                Vr = sparse_gauss_newton_FixMaterial_OptP(
                    geometry, F, targetV, Vr, P,
                    lambda_pf_s, kappa_pf_s,
                    masses, M_P_2, L_P_2, P_anchor,
                    kappa_reg + lambda_reg,
                    adjointFunc_OptP, fixedIdx,
                    config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                    wM_P, wL_P,
                    E, nu, ac.thickness,
                    config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                    distance, spn_energy, P_reg,
                    logger_OptP);

                // v7 flow: do NOT hard-snap (lambda, kappa) here.  The BCD
                // stages run entirely in the continuous space; one final
                // snap-material OptP after the stage loop refines P for the
                // actually-manufactured discrete design.
                MrInv = precomputeMrInv(mesh, P, F);
                M_kappa = computeFaceMassKappa(mesh, MrInv);
                distance = recomputeForwardState();
                kappa_reg  = computeKappaReg();
                lambda_reg = computeLambdaReg();
                spn_energy = distance + kappa_reg + lambda_reg;
                const auto _proj_st = computeProjStateFrom(Vr);
                const double pd_end  = _proj_st.dist;
                const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(_proj_st.V);
                const double pen_kap = penalty_to_kapp.eval(kappa_pf_s.toVector());
                const double pen_lam = penalty_to_lamb.eval(lambda_pf_s.toVector());
                spdlog::info("Patch {} Stage {} [OptP   finish] SPN={:.6f} Dist={:.6f} ProjDist={:.6f} bd_rms={:.6f} int_rms={:.6f}",
                             pd.idx, k, spn_energy, distance, pd_end, bd_rms, int_rms);
                iter_log_ofs << k << ",OptP,-1," << spn_energy << "," << distance << "," << pd_end << ","
                             << kappa_reg << "," << lambda_reg << ","
                             << pen_kap << "," << pen_lam << ","
                             << wP_kap << "," << wP_lam << ","
                             << wM_kap << "," << wL_kap << "," << wM_lam << "," << wL_lam << "\n";
            }

            // -- Trust-region safeguard --
            //
            // Snapshot update is decoupled from accept/reject:
            //   * best snapshot updates iff proj_new < proj_best, so
            //     `proj_best` strictly tracks the historical minimum.
            //   * reject (revert all state + shrink wP growth) still uses
            //     the "both worsen" rule the user originally specified.
            //
            // Otherwise (dist improves but proj worsens) we ACCEPT the
            // step and let SGN keep advancing, but the best snapshot
            // intentionally does *not* drift to a worse-proj state.  This
            // way the post-loop restore guarantees the final manufacturing
            // distance equals the historical minimum proj.
            const double dist_new = distance;
            const double proj_new = computeProjectedDistance();

            const bool snapshot_improves = (proj_new < proj_best);
            if (snapshot_improves) {
                proj_best       = proj_new;
                dist_best       = dist_new;
                Vr_best         = Vr;
                lambda_pf_best  = lambda_pf_s;
                kappa_pf_best   = kappa_pf_s;
                P_best          = P;
                MrInv_best      = MrInv;
                M_kappa_best    = M_kappa;
                wP_lam_best     = wP_lam;
                wP_kap_best     = wP_kap;
                kappa_reg_best  = kappa_reg;
                lambda_reg_best = lambda_reg;
            }

            const bool reject = (dist_new > dist_best) && (proj_new > proj_best);
            if (reject) {
                Vr          = Vr_best;
                lambda_pf_s = lambda_pf_best;
                kappa_pf_s  = kappa_pf_best;
                P           = P_best;
                MrInv       = MrInv_best;
                M_kappa     = M_kappa_best;
                // wM / wL keep their schedule-decayed values; not rolled back.
                wP_lam      = wP_lam_best;
                wP_kap      = wP_kap_best;
                kappa_reg   = kappa_reg_best;
                lambda_reg  = lambda_reg_best;
                wP_lam_growth_factor = std::max(1e-4, wP_lam_growth_factor * 0.5);
                wP_kap_growth_factor = std::max(1e-4, wP_kap_growth_factor * 0.5);
                spdlog::info("Patch {} Stage {} [REJECT] dist={:.6f}>{:.6f} AND proj={:.6f}>{:.6f}; revert, wP_lam_growth -> {:.6f}, wP_kap_growth -> {:.6f}",
                             pd.idx, k, dist_new, dist_best, proj_new, proj_best,
                             wP_lam_growth_factor, wP_kap_growth_factor);
            } else if (snapshot_improves) {
                spdlog::info("Patch {} Stage {} [ACCEPT, best updated] dist={:.6f} proj={:.6f} (best now)",
                             pd.idx, k, dist_new, proj_new);
            } else {
                spdlog::info("Patch {} Stage {} [ACCEPT, best unchanged] dist={:.6f} proj={:.6f} (best proj still {:.6f})",
                             pd.idx, k, dist_new, proj_new, proj_best);
            }

            k++;
            // Unconditional growth; safeguard handles overshoot.
            wP_lam *= (1.0 + wP_lam_growth_factor);
            wP_kap *= (1.0 + wP_kap_growth_factor);

            wM_kap *= 0.5;  wL_kap *= 0.5;
            wM_lam *= 0.5;  wL_lam *= 0.5;
            kappa_reg  = computeKappaReg();
            lambda_reg = computeLambdaReg();
        }

        // After the stage loop, restore the best snapshot so the downstream
        // material output / proj.obj is guaranteed to reflect the historical
        // minimum projected distance, not the last (possibly worse) state.
        Vr          = Vr_best;
        lambda_pf_s = lambda_pf_best;
        kappa_pf_s  = kappa_pf_best;
        P           = P_best;
        MrInv       = MrInv_best;
        M_kappa     = M_kappa_best;
        spdlog::info("Patch {} restored best snapshot: dist={:.6f}  proj={:.6f}",
                     pd.idx, dist_best, proj_best);

        // ---- Final snap-material OptP --------------------------------------
        // BCD loop optimised everything in continuous space; one extra SGN
        // OptP on snapped (= actually manufacturable) material refines P so
        // the manufactured forward-sim lands as close to V_T as possible.
        spdlog::info("Patch {} Final SNAP OptP", pd.idx);
        {
            FaceData<double> lambda_pf_snap(mesh);
            FaceData<double> kappa_pf_snap(mesh);
            for (Face f : mesh.faces()) {
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb,
                                            kappa_pf_s[f], lambda_pf_s[f]);
                lambda_pf_snap[f] = ac.feasible_lamb[idx];
                kappa_pf_snap[f]  = ac.feasible_kapp[idx];
            }
            auto adjointFunc_OptP_snap = adjointFunction_FixMaterial_OptP(
                geometry, F, lambda_pf_snap, kappa_pf_snap,
                E, nu, ac.thickness,
                config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);

            auto logger_OptP_snap = [&](int i, const Eigen::VectorXd& x_iter,
                                        double spn, double dist, double, double) {
                const auto _st = computeProjStateFrom(reshape_x_to_V(x_iter));
                const auto [bd_rms, int_rms] = computeBoundaryInteriorRMS(_st.V);
                const double pen_kap = penalty_to_kapp.eval(kappa_pf_s.toVector());
                const double pen_lam = penalty_to_lamb.eval(lambda_pf_s.toVector());
                std::cout << "\tdist=" << dist << "\tbd=" << bd_rms << "\tint=" << int_rms << std::endl;
                iter_log_ofs << "-1,FinalSnapOptP," << i << ","
                             << spn << "," << dist << "," << dist << ","
                             << kappa_reg << "," << lambda_reg << ","
                             << pen_kap << "," << pen_lam << ","
                             << wP_kap << "," << wP_lam << ","
                             << wM_kap << "," << wL_kap << "," << wM_lam << "," << wL_lam << ","
                             << bd_rms << "," << int_rms << "\n";
            };

            double dummy_dist = 0, dummy_spn = 0, dummy_reg = 0;
            Vr = sparse_gauss_newton_FixMaterial_OptP(
                geometry, F, targetV, Vr, P,
                lambda_pf_snap, kappa_pf_snap,
                masses, M_P_2, L_P_2, P_anchor,
                0.0,
                adjointFunc_OptP_snap, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
                config.RuntimeSetting.wM_P, config.RuntimeSetting.wL_P,
                E, nu, ac.thickness,
                config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces,
                dummy_dist, dummy_spn, dummy_reg,
                logger_OptP_snap);
            MrInv = precomputeMrInv(mesh, P, F);
            spdlog::info("Patch {} Final SNAP OptP done: dist={:.6f}", pd.idx, dummy_dist);
        }


#else

        Vr = targetV;
        double distance_kap = 0.0, spn_kap = 0.0, self_reg_kap = 0.0;
        double distance_lam = 0.0, spn_lam = 0.0, self_reg_lam = 0.0;
        double kappa_reg_np  = computeKappaReg();
        double lambda_reg_np = computeLambdaReg();
        while(k < stage_iter)
        {
            spdlog::info("Patch {} Stage {}, OptKap start", pd.idx, k);

            auto adjointFunc_OptKap = adjointFunction_FixLam_OptKap(geometry, F, MrInv, lambda_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixLam_OptKap(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, lambda_reg_np,
                adjointFunc_OptKap, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, config.RuntimeSetting.wM_kap, config.RuntimeSetting.wL_kap,
                E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces,
                distance_kap, spn_kap, self_reg_kap);
            kappa_reg_np = self_reg_kap;

            spdlog::info("Patch {} Stage {}, OptKap finish - Distance: {:.6f}, SPN energy: {:.6f}", pd.idx, k, distance_kap, spn_kap);


            spdlog::info("Patch {} Stage {}, OptLam start", pd.idx, k);
            auto adjointFunc_OptLam = adjointFunction_FixKap_OptLam2(geometry, F, MrInv, kappa_pf_s, E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Vr = sparse_gauss_newton_FixKap_OptLam(geometry, targetV, Vr, MrInv, lambda_pf_s, kappa_pf_s, masses, kappa_reg_np,
                adjointFunc_OptLam, fixedIdx,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, config.RuntimeSetting.wM_lam, config.RuntimeSetting.wL_lam,
                E, nu, ac.thickness, config.RuntimeSetting.w_s,config.RuntimeSetting.w_b, ref_faces,
                distance_lam, spn_lam, self_reg_lam);
            lambda_reg_np = self_reg_lam;

            spdlog::info("Patch {} Stage {}, OptLam finish - Distance: {:.6f}, SPN energy: {:.6f}", pd.idx, k, distance_lam, spn_lam);


            k++;
        }


#endif



        // Vr lives in the same physical frame as V; write as-is.
        igl::writeOBJ(morph_dir + "patch_" + std::to_string(pd.idx) + "_inv.obj", Vr, F);

        // ---- Final projected forward sim (manufacturing reality) ----------
        // Snap (lambda, kappa) per face to the nearest feasible (t1, t2) pair,
        // run forward Newton on the snapped material, write the resulting mesh
        // and report the manufacturing distance prominently.
        double final_proj_dist = 0.0;
        double final_bd_rms    = 0.0;
        double final_int_rms   = 0.0;
        {
            FaceData<double> kappa_pf_proj(mesh);
            FaceData<double> lambda_pf_proj(mesh);
            for (Face f : mesh.faces()) {
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb,
                                            kappa_pf_s[f], lambda_pf_s[f]);
                kappa_pf_proj[f]  = ac.feasible_kapp[idx];
                lambda_pf_proj[f] = ac.feasible_lamb[idx];
            }
            auto simFunc_proj = simulationFunction(geometry, MrInv, lambda_pf_proj, kappa_pf_proj,
                E, nu, ac.thickness, config.RuntimeSetting.w_s, config.RuntimeSetting.w_b, ref_faces);
            Eigen::MatrixXd Vr_proj = Vr;
            newton(geometry, Vr_proj, simFunc_proj,
                config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon, false, fixedIdx);
            for (size_t i = 0; i < nV; ++i)
                for (int j = 0; j < 3; ++j) {
                    double d = Vr_proj(i, j) - targetV(i, j);
                    final_proj_dist += masses(3 * i + j) * d * d;
                }
            // Per-vertex Euclidean RMS on the manufactured proj mesh.
            auto [_bd, _int] = computeBoundaryInteriorRMS(Vr_proj);
            final_bd_rms  = _bd;
            final_int_rms = _int;
            const std::string proj_path = morph_dir + "patch_" + std::to_string(pd.idx) + "_proj.obj";
            igl::writeOBJ(proj_path, Vr_proj, F);
            spdlog::info("Patch {} proj mesh -> {}", pd.idx, proj_path);
        }

        // ---- Material projection: per-face (lambda, kappa) -> nearest feasible (t1, t2) ----
        // Writes two files:
        //   patch_{i}_material.txt : face_id  t1  t2          (discrete grayscale)
        //   patch_{i}_lamkap.txt   : face_id  lambda  kappa   (continuous SGN values)
        {
            const std::string mat_path = design_dir + "patch_" + std::to_string(pd.idx) + "_material.txt";
            const std::string lk_path  = design_dir + "patch_" + std::to_string(pd.idx) + "_lamkap.txt";
            std::ofstream mof(mat_path);
            std::ofstream lof(lk_path);
            mof << "# face_id  t1  t2\n";
            lof << "# face_id  lambda  kappa\n";
            for (Face f : mesh.faces()) {
                double kap = kappa_pf_s[f];
                double lam = lambda_pf_s[f];
                int idx = find_feasible_idx(ac.feasible_kapp, ac.feasible_lamb, kap, lam);
                double t1 = ac.feasible_t_vals[idx].first;
                double t2 = ac.feasible_t_vals[idx].second;
                mof << f.getIndex() << "  " << t1  << "  " << t2  << "\n";
                lof << f.getIndex() << "  " << lam << "  " << kap << "\n";
            }
            spdlog::info("Patch {} material -> {}", pd.idx, mat_path);
            spdlog::info("Patch {} lamkap   -> {}", pd.idx, lk_path);
        }

        // Highlight: final manufacturing distance for this patch.
        std::cout << "\n";
        std::cout << "==========================================================\n";
        std::cout << "  Patch " << pd.idx
                  << "  FINAL Projected distance (manufactured): "
                  << final_proj_dist << "\n";
        std::cout << "  Per-vertex RMS (Euclidean, mm)  bd = " << final_bd_rms
                  << "   int = " << final_int_rms << "\n";
        std::cout << "==========================================================\n";
        std::cout << "\n";

        spdlog::info("Patch {} done.", pd.idx);
    }

    spdlog::info("All patches processed; program finish.");

}

