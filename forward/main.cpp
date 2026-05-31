// Forward: predict the deformed shape from a given material assignment.
//
// Reads (single-mesh / patch_0):
//   PathSetting.TargetDir + {model}/patch_0_V.obj             (target V — used as Newton init)
//   PathSetting.MorphInitDir + {model}/patch_0_P.obj          (OptP-optimized P from Inverse)
//   PathSetting.CondDir   + {model}/patch_0_bound_center.txt  (3 vertex idx — rigid anchor)
//   PathSetting.DesignDir + {model}/patch_0_material.txt      (per-face t1, t2)
//
// Writes:
//   PathSetting.ForwardDir + {model}/patch_0_pred.obj         (predicted deformed mesh)
//
// V_target is already in physical (device) units; the predicted mesh is in
// the same frame so it can be diff'd directly against the target.
// Run Param + (Inverse or any tool that produces the per-face material) first.

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

#include <spdlog/spdlog.h>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include "config.hpp"
#include "material.hpp"
#include "parameterization.h"
#include "simulation_utils.h"
#include "functions.h"
#include "newton.h"
#include "morphmesh.hpp"
#include "morph_functions.hpp"
#include "boundary_utils.h"

int main(int /*argc*/, char* /*argv*/[])
{
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    Config config("cfg.json");
    ActiveComposite ac(config.materialJsonPath());
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    spdlog::info("Forward (single mesh): start.");

    const std::string model           = config.ModelSetting.ModelName;
    const std::string target_dir      = config.PathSetting.TargetDir     + model + "/";
    const std::string morph_init_dir  = config.PathSetting.MorphInitDir  + model + "/";
    const std::string cond_dir        = config.PathSetting.CondDir       + model + "/";
    const std::string design_dir      = config.PathSetting.DesignDir     + model + "/";
    const std::string pred_dir        = config.PathSetting.ForwardDir    + model + "/";
    std::filesystem::create_directories(pred_dir);

    // -------- Load V (physical-unit target, also used as Newton init) --------
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    const std::string v_path = target_dir + "patch_0_V.obj";
    if (!igl::readOBJ(v_path, V, F)) {
        spdlog::error("Cannot read V: {}.  Run Param first.", v_path);
        return -1;
    }
    const Eigen::Index nV = V.rows();
    const Eigen::Index nF = F.rows();
    spdlog::info("V: {} V, {} F (from {}).", nV, nF, v_path);

    // -------- Load P --------
    // Forward MUST use the OptP-optimized P from Inverse (MorphInitDir), not
    // the initial Param P.  Using Param P silently would produce a "valid-looking"
    // verification that disagrees with the actual inverse design — same hazard
    // as S3_Simulate / S4_Slice.  Hard error if MorphInitDir copy is missing.
    Eigen::MatrixXd P_loaded3;
    Eigen::MatrixXi F_p;
    const std::string p_path = morph_init_dir + "patch_0_P.obj";
    if (!igl::readOBJ(p_path, P_loaded3, F_p)) {
        spdlog::error("Cannot read P: {}", p_path);
        spdlog::error("  -> Run Inverse on model '{}' first to produce the OptP-optimized P in MorphInitDir.", model);
        return -1;
    }
    spdlog::info("P loaded from MorphInitDir: {}", p_path);
    if (P_loaded3.rows() != nV) {
        spdlog::error("P vertex count ({}) does not match V count ({}).", P_loaded3.rows(), nV);
        return -1;
    }
    Eigen::MatrixXd P = P_loaded3.leftCols(2);

    // -------- Load cond (3 vertex idx -> 9 DOF) --------
    std::vector<int> fixedVertexIdx;
    {
        const std::string cond_path = cond_dir + "patch_0_bound_center.txt";
        std::ifstream ifs(cond_path);
        if (!ifs.is_open()) {
            spdlog::error("Cannot read cond: {}.  Run Param first.", cond_path);
            return -1;
        }
        int v0, v1, v2;
        ifs >> v0 >> v1 >> v2;
        fixedVertexIdx = {v0, v1, v2};
        spdlog::info("Loaded cond: v0={} v1={} v2={}", v0, v1, v2);
    }
    std::vector<int> fixedIdx;
    for (int v : fixedVertexIdx)
        for (int k = 0; k < 3; ++k) fixedIdx.push_back(3 * v + k);
    std::sort(fixedIdx.begin(), fixedIdx.end());

    // -------- Load per-face material (t1, t2) --------
    // If cfg.json sets Model.DesignName, use it (e.g. "hemisphere_material");
    // otherwise fall back to the per-patch convention.
    const std::string design_name = config.ModelSetting.DesignName.empty()
        ? std::string("patch_0_material")
        : config.ModelSetting.DesignName;
    Eigen::VectorXd t1_pf = Eigen::VectorXd::Zero(nF);
    Eigen::VectorXd t2_pf = Eigen::VectorXd::Zero(nF);
    {
        const std::string mat_path = design_dir + design_name + ".txt";
        std::ifstream ifs(mat_path);
        if (!ifs.is_open()) {
            spdlog::error("Cannot read material: {}.  Run Inverse first.", mat_path);
            return -1;
        }
        std::string line;
        Eigen::Index seen = 0;
        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            int fid; double t1, t2;
            if (!(iss >> fid >> t1 >> t2)) continue;
            if (fid < 0 || fid >= nF) {
                spdlog::error("Material line has bad face_id {} (nF={}).", fid, nF);
                return -1;
            }
            t1_pf(fid) = t1;
            t2_pf(fid) = t2;
            ++seen;
        }
        if (seen != nF) {
            spdlog::error("Material has {} face rows, mesh has {} faces.", seen, nF);
            return -1;
        }
        spdlog::info("Loaded material: {} faces.", seen);
    }

    // -------- Build mesh + geometry + reference structures --------
    ManifoldSurfaceMesh mesh(F);
    VertexPositionGeometry geometry(mesh, V);
    geometry.refreshQuantities();

    std::vector<bool> is_boundary_face;
    std::vector<int> ref_faces = buildRefFaces(mesh, is_boundary_face);
    FaceData<Eigen::Matrix2d> MrInv = precomputeMrInv(mesh, P, F);

    // -------- Convert per-face (t1, t2) -> per-face (lambda, kappa) --------
    FaceData<double> lam_pf(mesh);
    FaceData<double> kap_pf(mesh);
    for (Face f : mesh.faces()) {
        const int i = f.getIndex();
        const double t1 = t1_pf(i);
        const double t2 = t2_pf(i);
        lam_pf[f] = compute_lamb_d(ac.m_strain_curve, ac.m_moduls_curve, t1, t2);
        kap_pf[f] = compute_curv_d(ac.m_strain_curve, ac.m_moduls_curve, ac.thickness, t1, t2, ac.kappa_factor);
    }

    // -------- Forward elastic simulation: minimise W(V; lam, kap) over V --------
    const double nu = 0.5;
    auto simFunc = simulationFunction(geometry, MrInv, lam_pf, kap_pf,
        ac.m_E_surface, nu, ac.thickness,
        config.RuntimeSetting.w_s, config.RuntimeSetting.w_b,
        ref_faces);

    Eigen::MatrixXd Vr = V;          // Newton init = target (close to local minimum)
    spdlog::info("Step: forward newton.");
    newton(geometry, Vr, simFunc,
           config.RuntimeSetting.MaxIter, config.RuntimeSetting.epsilon,
           false, fixedIdx);

    // -------- Write predicted shape (named after the design for traceability) --------
    const std::string pred_path = pred_dir + design_name + "_pred.obj";
    igl::writeOBJ(pred_path, Vr, F);
    spdlog::info("Pred -> {}", pred_path);

    spdlog::info("Forward (single mesh): done.");
    return 0;
}
