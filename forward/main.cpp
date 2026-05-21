// Forward: predict the deformed shape from a given material assignment.
//
// Reads (single-mesh / patch_0):
//   PathSetting.TargetDir + {model}/patch_0_V.obj             (target V — used as Newton init)
//   PathSetting.ParamDir  + {model}/patch_0_P.obj             (scaled + gauge-shifted P)
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

    const std::string model      = config.ModelSetting.ModelName;
    const std::string target_dir = config.PathSetting.TargetDir  + model + "/";
    const std::string param_dir  = config.PathSetting.ParamDir   + model + "/";
    const std::string cond_dir   = config.PathSetting.CondDir    + model + "/";
    const std::string design_dir = config.PathSetting.DesignDir  + model + "/";
    const std::string pred_dir   = config.PathSetting.ForwardDir + model + "/";
    std::filesystem::create_directories(pred_dir);

    // File-name prefix selector. Legacy EvolutionCut pipeline uses patch_0_*;
    // ShapeGen synthetic-shape pipeline uses {model}_* to keep one name per
    // shape end-to-end (mesh, BC, design, pred all share the same model name).
    const bool whole_mesh = config.RuntimeSetting.whole_mesh_mode;
    const std::string p_name   = whole_mesh ? (model + "_param")        : "patch_0_P";
    const std::string c_name   = whole_mesh ? (model + "_bound_center") : "patch_0_bound_center";
    const std::string v_name   = whole_mesh ? (model + "_V")            : "patch_0_V";

    // -------- Load P (2D parameterization, always required) --------
    Eigen::MatrixXd P_loaded3;
    Eigen::MatrixXi F_p;
    const std::string p_path = param_dir + p_name + ".obj";
    if (!igl::readOBJ(p_path, P_loaded3, F_p)) {
        spdlog::error("Cannot read P: {}.  Run Param first.", p_path);
        return -1;
    }
    const Eigen::Index nV = P_loaded3.rows();
    const Eigen::Index nF = F_p.rows();
    spdlog::info("P: {} V, {} F (from {}).", nV, nF, p_path);
    Eigen::MatrixXd P = P_loaded3.leftCols(2);

    // -------- Load or synthesize V (Newton init) --------
    // Default behavior (init_from_param=false): read the target 3D mesh
    // from target_dir, both as the Newton init AND as the reference deformed
    // geometry the design was solved against -- this is the standard
    // EvolutionCut → Inverse → Forward path.
    //
    // Forward-only behavior (init_from_param=true): we have no target;
    // initialize V = [P, eps*N(0,1)] so Newton starts at a near-flat plate
    // with a tiny z perturbation to break the planar-saddle degeneracy.
    // The strain-driven gradient at the flat state then pushes V away from
    // z=0 toward the curved minimum determined by the design.
    Eigen::MatrixXd V;
    Eigen::MatrixXi F = F_p;
    if (config.RuntimeSetting.init_from_param) {
        constexpr double kInitZNoise = 1e-3;
        V = Eigen::MatrixXd::Zero(nV, 3);
        V.leftCols(2) = P;
        std::srand(0);
        for (Eigen::Index i = 0; i < nV; ++i) {
            V(i, 2) = kInitZNoise * ((std::rand() / double(RAND_MAX)) * 2.0 - 1.0);
        }
        spdlog::info("V: {} V, {} F (synthesized from P + N(0,{:.0e}) z noise, init_from_param=true).",
                     nV, nF, kInitZNoise);
    } else {
        const std::string v_path = target_dir + v_name + ".obj";
        Eigen::MatrixXi F_v;
        if (!igl::readOBJ(v_path, V, F_v)) {
            spdlog::error("Cannot read V: {}.  Run Param first or set RuntimeSettings.InitFromParam=true.", v_path);
            return -1;
        }
        if (V.rows() != nV) {
            spdlog::error("V vertex count ({}) does not match P count ({}).", V.rows(), nV);
            return -1;
        }
        F = F_v;
        spdlog::info("V: {} V, {} F (target loaded from {}).", nV, nF, v_path);
    }

    // -------- Load cond (3 vertex idx -> 9 DOF) --------
    std::vector<int> fixedVertexIdx;
    {
        const std::string cond_path = cond_dir + c_name + ".txt";
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
        lam_pf[f] = compute_lamb_d<double>(ac.m_strain_curve, t1, t2);
        kap_pf[f] = compute_curv_d<double>(ac.m_strain_curve, ac.thickness, t1, t2);
    }

    // -------- Forward elastic simulation: minimise W(V; lam, kap) over V --------
    const double E  = 1.0;
    const double nu = 0.5;
    auto simFunc = simulationFunction(geometry, MrInv, lam_pf, kap_pf,
        E, nu, ac.thickness,
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
