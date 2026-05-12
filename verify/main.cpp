/// Verify -- inverse design with a pre-computed parameterization
/// (twin-experiment / round-trip validation).
///
/// Differs from Inverse in one key respect: rather than running
/// parameterizeMesh() on the input target, Verify reads a pre-computed flat
/// parameterization (_param.obj) and a 3D target with matching topology,
/// then invokes runInverseDesign() directly.  This is the exact-twin
/// (inverse-crime) test path -- the recovered material can be compared
/// element-wise against the ground truth used to produce the target via
/// Forward.
///
/// Inputs (resolved from verify_cfg.json):
///   param mesh:  {param_path}/{name}/{name}_param.obj         (2D, fixed)
///   target obj:  {target_path}/{name}/{design_name}_forward.obj
///   BC file:     {cond_path}/{name}/bound_center.txt          (3 vertex ids)
///   curves:      {material.curves_path}
///
/// Outputs:
///   {morph_path}/{name}/{design_name}_target.obj   -- V scaled to platewidth
///   {morph_path}/{name}/{design_name}_inv.obj      -- continuous SGN result
///   {morph_path}/{name}/{design_name}_proj.obj     -- after material projection
///   {morph_path}/{name}/{design_name}_metrics.txt  -- per-face feasibility excess
///   {design_path}/{name}/{design_name}_recovered.txt  -- recovered (t1, t2)
///
/// Usage:
///   ./Verify                       (reads ./verify_cfg.json)
///   ./Verify <path/to/cfg.json>

#include <Eigen/Core>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>
#include <nlohmann/json.hpp>

#include "inverse_design.h"
#include "material.hpp"
#include "morph_functions.hpp"
#include "morphmesh.hpp"
#include "setup.hpp"
#include "simulation_utils.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace {

/// PCA-rotate the 2D parameterization so its principal axis aligns with x.
/// Mirrors the helper of the same name in inverse/main.cpp.
std::pair<Eigen::MatrixXd, double> pcaAlignP(const Eigen::MatrixXd& P)
{
    Eigen::RowVector2d center = P.colwise().mean();
    Eigen::MatrixXd Pc = P.rowwise() - center;

    Eigen::Matrix2d C = Pc.transpose() * Pc / static_cast<double>(Pc.rows());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(C);

    Eigen::Matrix2d Q = eig.eigenvectors();
    if (eig.eigenvalues()(0) > eig.eigenvalues()(1))
        Q.col(0).swap(Q.col(1));
    if (Q.determinant() < 0.0)
        Q.col(1) *= -1.0;

    Eigen::MatrixXd P_rot = Pc * Q;
    double xExtent = P_rot.col(0).maxCoeff() - P_rot.col(0).minCoeff();
    return {P_rot, xExtent};
}

/// Read whitespace-separated vertex indices from a cond file and expand each
/// to 3 DOF indices (x, y, z).
std::vector<int> readCondFile(const std::string& path)
{
    std::ifstream ifs(path);
    if (!ifs)
        return {};
    std::vector<int> vertIds;
    for (int v; ifs >> v;)
        vertIds.push_back(v);

    std::vector<int> fixedIdx;
    fixedIdx.reserve(vertIds.size() * 3);
    for (int vid : vertIds)
        for (int d = 0; d < 3; ++d)
            fixedIdx.push_back(vid * 3 + d);
    std::sort(fixedIdx.begin(), fixedIdx.end());
    return fixedIdx;
}

std::string withSlash(std::string s)
{
    if (!s.empty() && s.back() != '/') s += '/';
    return s;
}

/// Parse a design file with lines: `face_id  t1  t2` (`#` comments allowed).
/// Mirrors `loadMaterial` in forward/main.cpp -- duplicated locally to keep
/// Verify a self-contained twin-experiment binary without dragging in the
/// Forward translation unit.
std::pair<Eigen::VectorXd, Eigen::VectorXd>
loadDesignFile(const std::string& path, int nF)
{
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("Could not open design file: " + path);

    Eigen::VectorXd t1 = Eigen::VectorXd::Zero(nF);
    Eigen::VectorXd t2 = Eigen::VectorXd::Zero(nF);
    std::vector<bool> seen(nF, false);

    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        if (const auto pos = line.find('#'); pos != std::string::npos)
            line.erase(pos);
        if (line.find_first_not_of(" \t\r\n") == std::string::npos)
            continue;

        std::istringstream iss(line);
        int    fid = -1;
        double v1  = 0.0;
        double v2  = 0.0;
        if (!(iss >> fid >> v1 >> v2))
            throw std::runtime_error(
                "Malformed line " + std::to_string(lineNo) + " in " + path);
        if (fid < 0 || fid >= nF)
            throw std::runtime_error(
                "Face id " + std::to_string(fid) + " out of range [0, "
                + std::to_string(nF) + ") in " + path);
        seen[fid] = true;
        t1[fid]   = v1;
        t2[fid]   = v2;
    }

    for (int i = 0; i < nF; ++i)
        if (!seen[i])
            throw std::runtime_error(
                "Missing material for face " + std::to_string(i) + " in " + path);

    return {std::move(t1), std::move(t2)};
}

} // anonymous namespace

int main(int argc, char* argv[])
{
    const std::string cfgPath = (argc >= 2) ? argv[1] : "verify_cfg.json";
    const std::string cfgAbs  = std::filesystem::weakly_canonical(cfgPath).string();
    spdlog::info("Reading config: {} (resolved: {})", cfgPath, cfgAbs);
    spdlog::info("Current working directory: {}",
                 std::filesystem::current_path().string());

    if (!std::filesystem::exists(cfgPath)) {
        spdlog::error("Config not found: {}", cfgAbs);
        spdlog::error("Usage: {} [path/to/verify_cfg.json]",
                      argc > 0 ? argv[0] : "Verify");
        return -1;
    }

    nlohmann::json cfg;
    try {
        std::ifstream ifs(cfgPath);
        ifs >> cfg;
    } catch (const std::exception& e) {
        spdlog::error("Failed to parse JSON from: {}", cfgAbs);
        spdlog::error("Parser error: {}", e.what());
        // Read and show first 200 bytes so user can see what was really opened.
        std::ifstream ifs(cfgPath, std::ios::binary);
        std::string head(200, '\0');
        ifs.read(head.data(), head.size());
        head.resize(ifs.gcount());
        spdlog::error("First {} bytes of file:\n{}", head.size(), head);
        return -1;
    }

    // -- Parse config ---------------------------------------------------------
    const std::string name       = cfg.at("model").at("name").get<std::string>();
    const std::string designName = cfg.at("design").at("design_name").get<std::string>();

    const std::string paramRoot   = withSlash(cfg.at("design").value("param_path",
                                              std::string("../Resources/param/")));
    const std::string targetRoot  = withSlash(cfg.at("design").value("target_path",
                                              std::string("../Resources/forward/")));
    const std::string condRoot    = withSlash(cfg.at("design").value("cond_path",
                                              std::string("../Resources/cond/")));
    const std::string designRoot  = withSlash(cfg.at("design").value("design_path",
                                              std::string("../Resources/design/")));
    const std::string morphRoot   = withSlash(cfg.at("design").value("morph_path",
                                              std::string("../Resources/morph/")));
    /// Trajectory CSV output directory.  Empty -> trajectory logging is off.
    /// Lives in `design` (output paths) section, not `solver` (algorithm
    /// parameters).
    const std::string traj_dir    = cfg.at("design").value("trajectory_dir",
                                              std::string());
    const std::string traj_tag    = cfg.at("design").value("trajectory_tag",
                                              std::string());

    const std::string curvesPath = cfg.at("material").at("curves_path").get<std::string>();

    // Load global setup (Resources/setup/global.json) before reading solver
    // section so that scalars default to setup values; cfg.solver may still
    // override per-experiment.
    LoadGlobalSetupOptions sopts;
    sopts.cfg_path = std::filesystem::path(cfgPath);
    sopts.material_path = std::filesystem::path(curvesPath);
    const GlobalSetup setup = loadGlobalSetup(cfg, sopts);

    const auto& sol = cfg.at("solver");
    double platewidth     = setup.platewidth;
    double poisson_ratio  = setup.poisson_ratio;
    if (sol.contains("platewidth")) {
        const double v = sol.value("platewidth", platewidth);
        spdlog::warn("Verify: cfg.solver.platewidth={} overrides setup.platewidth={}",
                     v, setup.platewidth);
        platewidth = v;
    }
    if (sol.contains("poisson_ratio")) {
        const double v = sol.value("poisson_ratio", poisson_ratio);
        spdlog::warn("Verify: cfg.solver.poisson_ratio={} overrides setup.poisson_ratio={}",
                     v, setup.poisson_ratio);
        poisson_ratio = v;
    }
    const int    max_iter          = sol.value("max_iter", 20);
    const double epsilon           = sol.value("epsilon", 1e-6);
    const double w_s               = sol.value("w_s", 1.0);
    const double w_b               = sol.value("w_b", 1.0);
    const double wM_kap            = sol.value("wM_kap", 0.1);
    const double wL_kap            = sol.value("wL_kap", 0.1);
    const double wM_lam            = sol.value("wM_lam", 0.0);
    const double wL_lam            = sol.value("wL_lam", 0.1);
    const double wP_kap            = sol.value("wP_kap", 0.01);
    const double wP_lam            = sol.value("wP_lam", 0.00);
    const double penalty_threshold = sol.value("penalty_threshold", 0.01);
    // well_scale: Lorentzian-soft-min sharpness for the energy-weighted joint
    // material penalty (replaces old log-sum-exp `betaP`).  Default 30000 gives
    // wells ~grid_spacing/3 in energy distance for the default material window.
    const double well_scale        = sol.value("well_scale", sol.value("betaP", 30000.0));
    const double wP_growth_factor  = sol.value("wP_growth_factor", 5.0);
    const double wM_decay_factor   = sol.value("wM_decay_factor", 1.0);
    const double wL_decay_factor   = sol.value("wL_decay_factor", 1.0);
    const int    verify_max_iter   = sol.value("verify_max_iter", 100);
    const double verify_epsilon    = sol.value("verify_epsilon", 1e-6);
    const int    max_stages        = sol.value("max_stages", 5);
    /// Twin-experiment oracle: when true, Verify reads the original
    /// {designRoot}/{name}/{designName}.txt that produced the target via
    /// Forward, recomputes the per-face (lambda, kappa) by polynomial
    /// evaluation at the exact (t1, t2), and feeds them to runInverseDesign
    /// instead of the Morphophing-from-V derivation.  A correct pipeline
    /// should then return dist_proj ~ 0 immediately.
    const bool   oracle_init       = sol.value("oracle_init", false);

    // -- Resolved file paths --------------------------------------------------
    const std::string paramFile  = paramRoot  + name + "/" + name + "_param.obj";
    const std::string targetFile = targetRoot + name + "/" + designName + "_forward.obj";
    const std::string condFile   = condRoot   + name + "/bound_center.txt";
    const std::string outMorph   = morphRoot  + name + "/";
    const std::string outDesign  = designRoot + name + "/";

    spdlog::info("Verify config:");
    spdlog::info("  shape  = {}", name);
    spdlog::info("  design = {}", designName);
    spdlog::info("  param  = {}", paramFile);
    spdlog::info("  target = {}", targetFile);
    spdlog::info("  cond   = {}", condFile);

    // -- Material -------------------------------------------------------------
    ActiveComposite ac(curvesPath);
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();

    // -- Load meshes ----------------------------------------------------------
    Eigen::MatrixXd V_tgt, P_3d;
    Eigen::MatrixXi F_tgt, F_par;
    if (!igl::readOBJ(targetFile, V_tgt, F_tgt)) {
        spdlog::error("Cannot read target: {}", targetFile);
        return -1;
    }
    if (!igl::readOBJ(paramFile, P_3d, F_par)) {
        spdlog::error("Cannot read param:  {}", paramFile);
        return -1;
    }

    // Topology must match -- verify reads the SAME mesh that was forwarded.
    if (V_tgt.rows() != P_3d.rows() || F_tgt.rows() != F_par.rows()) {
        spdlog::error("Topology mismatch -- target ({} V, {} F) vs param ({} V, {} F)",
                      V_tgt.rows(), F_tgt.rows(), P_3d.rows(), F_par.rows());
        spdlog::error("Verify requires the target to be a Forward run on the same param mesh.");
        return -1;
    }
    if ((F_tgt - F_par).cwiseAbs().sum() != 0) {
        spdlog::error("Face arrays differ between target and param.");
        return -1;
    }

    // 2D P (drop the z column from the OBJ flat parameterization).
    Eigen::MatrixXd P_2d(P_3d.rows(), 2);
    P_2d.col(0) = P_3d.col(0);
    P_2d.col(1) = P_3d.col(1);

    // -- PCA align + global scale (mirrors inverse/main.cpp Phase 2) ---------
    //
    // For twin-experiment runs the GT material was filled on this exact param
    // mesh in its native coordinate system: any rescale changes kappa (which
    // has units 1/length) by 1/globalScale while ac.feasible_kapp stays in
    // original units, breaking the on-grid invariance that the twin test
    // relies on.  Setting globalScale = 1 keeps the comparison consistent.
    auto [P_rot, xExtent] = pcaAlignP(P_2d);
    const double globalScale = 1.0;

    Eigen::MatrixXd V_scaled = V_tgt * globalScale;
    Eigen::MatrixXd P_scaled = P_rot * globalScale;

    spdlog::info("globalScale = {:.6f}  (xExtent = {:.4f}, platewidth = {:.4f})",
                 globalScale, xExtent, platewidth);
    spdlog::info("Mesh: {} vertices, {} faces", V_scaled.rows(), F_par.rows());

    // -- Read BC (fixed vertex indices -> DOF indices) -----------------------
    std::vector<int> fixedIdx = readCondFile(condFile);
    if (fixedIdx.empty()) {
        spdlog::error("No fixed DOFs read from cond file: {}", condFile);
        return -1;
    }
    spdlog::info("Read {} fixed DOFs ({} vertices) from cond file",
                 fixedIdx.size(), fixedIdx.size() / 3);
    spdlog::info("Sanity: V dims=({}, {})  P dims=({}, {})  F dims=({}, {})",
                 V_scaled.rows(), V_scaled.cols(),
                 P_scaled.rows(), P_scaled.cols(),
                 F_par.rows(), F_par.cols());
    spdlog::info("Sanity: F max vert idx = {} (V has {} rows)",
                 F_par.maxCoeff(), V_scaled.rows());
    spdlog::info("Sanity: fixedIdx range = [{}, {}] (max DOF = {})",
                 fixedIdx.front(), fixedIdx.back(), 3 * V_scaled.rows() - 1);

    // -- Build geometry-central structures ----------------------------------
    ManifoldSurfaceMesh mesh(F_par);
    VertexPositionGeometry geom(mesh, V_scaled);
    geom.refreshQuantities();

    FaceData<Eigen::Matrix2d> MrInv = precomputeMrInv(mesh, P_scaled, F_par);

    // -- Run inverse design --------------------------------------------------
    InverseDesignProblem problem;
    problem.V                 = V_scaled;
    problem.F                 = F_par;
    problem.P                 = P_scaled;
    problem.mesh              = &mesh;
    problem.geometry          = &geom;
    problem.MrInv             = MrInv;
    problem.fixedIdx          = fixedIdx;
    problem.ac                = &ac;
    problem.poisson_ratio     = poisson_ratio;
    problem.max_iter          = max_iter;
    problem.epsilon           = epsilon;
    problem.w_s               = w_s;
    problem.w_b               = w_b;
    problem.wM_kap            = wM_kap;
    problem.wL_kap            = wL_kap;
    problem.wM_lam            = wM_lam;
    problem.wL_lam            = wL_lam;
    problem.wP_kap            = wP_kap;
    problem.wP_lam            = wP_lam;
    problem.penalty_threshold = penalty_threshold;
    problem.well_scale        = well_scale;
    problem.wP_growth_factor  = wP_growth_factor;
    problem.wM_decay_factor   = wM_decay_factor;
    problem.wL_decay_factor   = wL_decay_factor;
    problem.verify_max_iter   = verify_max_iter;
    problem.verify_epsilon    = verify_epsilon;
    problem.max_stages        = max_stages;
    problem.patch_id          = 0;
    problem.trajectory_dir    = traj_dir;
    problem.trajectory_tag    = traj_tag.empty() ? (name + "_" + designName) : traj_tag;

    // Twin-experiment oracle: replace the Morphophing-from-V starting metric
    // with the exact (lambda, kappa) recomputed from the original (t1, t2).
    if (oracle_init) {
        const std::string designFile = designRoot + name + "/" + designName + ".txt";
        spdlog::info("Oracle init enabled: reading reference (t1, t2) from {}", designFile);

        auto [t1, t2] = loadDesignFile(designFile, static_cast<int>(F_par.rows()));

        Eigen::VectorXd lam_pf = Eigen::VectorXd::Zero(F_par.rows());
        Eigen::VectorXd kap_pf = Eigen::VectorXd::Zero(F_par.rows());
        for (int f = 0; f < F_par.rows(); ++f) {
            lam_pf[f] = compute_lamb_d(ac.m_strain_curve, t1[f], t2[f]);
            kap_pf[f] = compute_curv_d(ac.m_strain_curve, ac.thickness, t1[f], t2[f]);
        }
        problem.oracle_lambda_pf = std::move(lam_pf);
        problem.oracle_kappa_pf  = std::move(kap_pf);
        spdlog::info(
            "Oracle (lambda, kappa) populated: nF={}, lambda in [{:.4f}, {:.4f}], "
            "kappa in [{:.4f}, {:.4f}]",
            problem.oracle_lambda_pf.size(),
            problem.oracle_lambda_pf.minCoeff(), problem.oracle_lambda_pf.maxCoeff(),
            problem.oracle_kappa_pf.minCoeff(),  problem.oracle_kappa_pf.maxCoeff());
    }

    spdlog::info("Schedule: wP_growth = {:.2f}, wM_decay = {:.2f}, wL_decay = {:.2f}",
                 wP_growth_factor, wM_decay_factor, wL_decay_factor);

    InverseDesignResult result = runInverseDesign(problem);

    // -- Write outputs --------------------------------------------------------
    std::filesystem::create_directories(outMorph);
    std::filesystem::create_directories(outDesign);

    igl::writeOBJ(outMorph + designName + "_target.obj", V_scaled,      F_par);
    igl::writeOBJ(outMorph + designName + "_inv.obj",    result.V_inv,  F_par);
    igl::writeOBJ(outMorph + designName + "_proj.obj",   result.V_proj, F_par);
    spdlog::info("Wrote target/inv/proj objs to {}", outMorph);

    {
        const std::string matPath = outDesign + designName + "_recovered.txt";
        std::ofstream ofs(matPath);
        ofs << "# face_id  t1  t2\n";
        for (int i = 0; i < F_par.rows(); ++i)
            ofs << i << "  " << result.t1[i] << "  " << result.t2[i] << "\n";
        spdlog::info("Recovered material -> {}", matPath);
    }

    {
        const std::string metPath = outMorph + designName + "_metrics.txt";
        std::ofstream ofs(metPath);
        ofs << "# face_id  lambda_excess  kappa_excess\n";
        for (int i = 0; i < F_par.rows(); ++i)
            ofs << i << "  " << result.lam_excess[i] << "  " << result.kap_excess[i] << "\n";
        spdlog::info("Metrics -> {}", metPath);
    }

    spdlog::info("Verify finished.  dist_inv={:.6f}  dist_proj={:.6f}",
                 result.dist_inv, result.dist_proj);
    return 0;
}
