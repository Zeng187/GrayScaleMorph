/// Standalone Forward simulation entry point.
///
/// Given a flat parameterized mesh (_param.obj), per-face material assignments
/// (_material.txt), and material polynomial curves (JSON), runs the Newton
/// forward simulation to recover the 3D equilibrium shape.
///
/// Usage:
///   # Explicit file mode
///   ./Forward --param <param.obj> --material <material.txt> \
///             --curves <curves.json> --output <out.obj>
///
///   # Config mode (derives paths from cfg.json)
///   ./Forward --cfg <cfg.json> [--output <out.obj>]

#include <Eigen/Core>

#include <igl/boundary_loop.h>
#include <igl/loop.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

#include <spdlog/spdlog.h>

#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "boundary_utils.h"
#include "config.hpp"
#include "functions.h"
#include "material.hpp"
#include "morphmesh.hpp"
#include "morph_functions.hpp"
#include "newton.h"
#include "output.hpp"
#include "simulation_utils.h"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

namespace {

/// Dimensionless reference stiffness (matches inverse_design.cpp).
constexpr double kE  = 1.0;
constexpr double kNu = 0.5;

/// Default energy weights for forward simulation.
constexpr double kWs = 1.0;
constexpr double kWb = 1.0;

// ---------------------------------------------------------------------------
// CLI types
// ---------------------------------------------------------------------------

struct CliOptions
{
    std::string cfgPath;
    std::string paramPath;
    std::string materialPath;
    std::string curvesPath;
    std::string outputPath;
};

/// Fully resolved paths + solver parameters for the simulation.
struct ForwardInputs
{
    std::string paramPath;
    std::string materialPath;
    std::string curvesPath;
    std::string outputPath;
    std::string condPath;      ///< boundary condition file (fixed vertex indices)
    int         maxIter = 20;
    double      epsilon = 1e-6;
    int         nfMin   = 0;   ///< minimum face count; Loop-subdivide if below
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string usageString(const char* prog)
{
    return "Usage:\n"
           "  " + std::string(prog) +
           " --param <p.obj> --material <m.txt> --curves <c.json> --output <o.obj>\n"
           "  " + std::string(prog) + " --cfg <cfg.json> [--output <o.obj>]";
}

/// Build a flat 3D position matrix from a 2D parameterization (z = 0).
Eigen::MatrixXd makeFlatFromParam(const Eigen::MatrixXd& P)
{
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(P.rows(), 3);
    V.col(0) = P.col(0);
    V.col(1) = P.col(1);
    return V;
}

/// Ensure the parent directory of `path` exists.
void ensureParentDir(const std::string& path)
{
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

CliOptions parseCli(int argc, char* argv[])
{
    if (argc < 2)
        throw std::runtime_error(usageString(argv[0]));

    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(
                    std::string("Missing value after ") + flag);
            return argv[++i];
        };

        if      (arg == "--cfg")      opts.cfgPath      = next("--cfg");
        else if (arg == "--param")    opts.paramPath     = next("--param");
        else if (arg == "--material") opts.materialPath  = next("--material");
        else if (arg == "--curves")   opts.curvesPath    = next("--curves");
        else if (arg == "--output")   opts.outputPath    = next("--output");
        else
            throw std::runtime_error("Unknown argument: " + arg + "\n"
                                     + usageString(argv[0]));
    }

    const bool cfgMode      = !opts.cfgPath.empty();
    const bool explicitMode = !opts.paramPath.empty()
                           || !opts.materialPath.empty()
                           || !opts.curvesPath.empty();

    if (cfgMode && explicitMode)
        throw std::runtime_error(
            "Cannot mix --cfg with --param/--material/--curves.\n"
            + usageString(argv[0]));
    if (!cfgMode && !explicitMode)
        throw std::runtime_error(usageString(argv[0]));

    // In explicit mode all four flags are mandatory.
    if (explicitMode) {
        if (opts.paramPath.empty() || opts.materialPath.empty()
            || opts.curvesPath.empty() || opts.outputPath.empty())
            throw std::runtime_error(
                "Explicit mode requires --param, --material, --curves, and --output.\n"
                + usageString(argv[0]));
    }

    return opts;
}

/// Resolve final file paths from either config JSON or explicit options.
///
/// The config JSON is parsed directly (not via the shared Config class)
/// because Forward has its own layout:
///
///   model.name           — model name (used for param file naming)
///   material.curves_path — material polynomial curves JSON
///   path.param_path      — directory containing {model}_param.obj
///   path.output_path     — directory for forward output
///   design.design_name   — design scheme name (for material file naming)
///   design.design_path   — directory containing {design_name}_material.txt
///   solver.max_iter      — Newton max iterations
///   solver.epsilon       — convergence tolerance
ForwardInputs resolveInputs(const CliOptions& opts)
{
    ForwardInputs inputs;

    if (!opts.cfgPath.empty()) {
        std::ifstream file(opts.cfgPath);
        if (!file.is_open())
            throw std::runtime_error("Cannot open config: " + opts.cfgPath);
        nlohmann::json j;
        file >> j;

        // model
        std::string modelName = j.at("model").at("name").get<std::string>();

        // material curves
        inputs.curvesPath = j.at("material").at("curves_path").get<std::string>();

        // paths: param input + forward output
        std::string paramPath  = j.at("path").value("param_path", "../Resources/param/");
        std::string outputPath = j.at("path").value("output_path", "../Resources/forward/");
        // Ensure trailing slash
        if (!paramPath.empty() && paramPath.back() != '/') paramPath += '/';
        if (!outputPath.empty() && outputPath.back() != '/') outputPath += '/';

        inputs.paramPath = paramPath + modelName + "/" + modelName + "_param.obj";

        // design: material input
        std::string designName = j.at("design").at("design_name").get<std::string>();
        std::string designPath = j.at("design").at("design_path").get<std::string>();
        if (!designPath.empty() && designPath.back() != '/') designPath += '/';

        inputs.materialPath = designPath + designName + ".txt";

        // output
        inputs.outputPath = !opts.outputPath.empty()
            ? opts.outputPath
            : outputPath + modelName + "/" + designName + "_forward.obj";

        // design: boundary condition
        if (j.at("design").contains("cond_name") && j.at("design").contains("cond_path")) {
            std::string condName = j["design"]["cond_name"].get<std::string>();
            std::string condPath = j["design"]["cond_path"].get<std::string>();
            if (!condPath.empty() && condPath.back() != '/') condPath += '/';
            inputs.condPath = condPath + condName;
        }

        // solver
        if (j.contains("solver")) {
            inputs.maxIter = j["solver"].value("max_iter", inputs.maxIter);
            inputs.epsilon = j["solver"].value("epsilon", inputs.epsilon);
            inputs.nfMin   = j["solver"].value("nf_min",  inputs.nfMin);
        }

        spdlog::info("Config loaded from: {}", opts.cfgPath);
    } else {
        inputs.paramPath    = opts.paramPath;
        inputs.materialPath = opts.materialPath;
        inputs.curvesPath   = opts.curvesPath;
        inputs.outputPath   = opts.outputPath;
    }

    return inputs;
}

// ---------------------------------------------------------------------------
// I/O: parameterized mesh
// ---------------------------------------------------------------------------

/// Load a 2D parameterized mesh (OBJ with z=0).
/// Returns the 2D coordinates (Nx2) and face indices.
struct ParamMesh
{
    Eigen::MatrixXd P;  ///< Nx2 parameterization coordinates
    Eigen::MatrixXi F;  ///< Triangles
};

ParamMesh loadParamMesh(const std::string& path)
{
    Eigen::MatrixXd V3;
    Eigen::MatrixXi F;
    if (!igl::readOBJ(path, V3, F))
        throw std::runtime_error("Could not read parameterized mesh: " + path);
    if (V3.rows() == 0 || F.rows() == 0)
        throw std::runtime_error("Parameterized mesh is empty: " + path);

    // Warn if z-coordinates are not all zero.
    if (V3.cols() >= 3) {
        const double maxZ = V3.col(2).array().abs().maxCoeff();
        if (maxZ > 1e-9)
            spdlog::warn("Param mesh has non-zero z (max |z| = {:.6e}). "
                         "Using x,y only.", maxZ);
    }

    return { V3.leftCols(2), std::move(F) };
}

// ---------------------------------------------------------------------------
// I/O: material assignment
// ---------------------------------------------------------------------------

/// Parse a material assignment file with lines: `face_id  t1  t2`.
/// Lines starting with '#' are treated as comments.
/// Returns per-face dose vectors (t1, t2), each of length nF.
std::pair<Eigen::VectorXd, Eigen::VectorXd>
loadMaterial(const std::string& path, int nF)
{
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("Could not open material file: " + path);

    Eigen::VectorXd t1 = Eigen::VectorXd::Zero(nF);
    Eigen::VectorXd t2 = Eigen::VectorXd::Zero(nF);
    std::vector<bool> seen(nF, false);

    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;

        // Strip comments and blank lines.
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
                + std::to_string(nF) + ") at line " + std::to_string(lineNo)
                + " in " + path);
        if (seen[fid])
            throw std::runtime_error(
                "Duplicate face id " + std::to_string(fid) + " at line "
                + std::to_string(lineNo) + " in " + path);

        seen[fid] = true;
        t1[fid]   = v1;
        t2[fid]   = v2;
    }

    for (int i = 0; i < nF; ++i) {
        if (!seen[i])
            throw std::runtime_error(
                "Missing material for face " + std::to_string(i) + " in " + path);
    }

    return { std::move(t1), std::move(t2) };
}

} // anonymous namespace

// ===========================================================================
// Entry point
// ===========================================================================

int main(int argc, char* argv[])
{
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    try {
        // -- CLI & path resolution ----------------------------------------
        const CliOptions    opts   = parseCli(argc, argv);
        const ForwardInputs inputs = resolveInputs(opts);

        spdlog::info("Forward: param    = {}", inputs.paramPath);
        spdlog::info("Forward: material = {}", inputs.materialPath);
        spdlog::info("Forward: curves   = {}", inputs.curvesPath);
        spdlog::info("Forward: output   = {}", inputs.outputPath);

        // ================================================================
        // Phase 1: Load all inputs
        // ================================================================

        // -- Load parameterized mesh --------------------------------------
        auto [P, F] = loadParamMesh(inputs.paramPath);
        spdlog::info("Loaded param mesh: {} vertices, {} faces.", P.rows(), F.rows());

        // -- Load per-face material assignments ---------------------------
        auto [t1, t2] = loadMaterial(inputs.materialPath, static_cast<int>(F.rows()));
        spdlog::info("Loaded material assignments for {} faces.", F.rows());

        // -- Load material curves -----------------------------------------
        ActiveComposite ac(inputs.curvesPath);
        ac.ComputeMaterialCurve();
        if (ac.thickness <= 0.0)
            throw std::runtime_error(
                "Material thickness must be positive (got "
                + std::to_string(ac.thickness) + ").");
        spdlog::info("Material curves loaded (thickness = {}).", ac.thickness);

        // -- Load boundary condition (fixed vertex indices) ---------------
        std::vector<int> fixedVertices;  // vertex indices to fix
        if (!inputs.condPath.empty()) {
            std::ifstream condFile(inputs.condPath);
            if (!condFile.is_open())
                throw std::runtime_error("Could not open boundary condition file: " + inputs.condPath);
            int vid;
            while (condFile >> vid)
                fixedVertices.push_back(vid);
            if (fixedVertices.empty())
                throw std::runtime_error("Boundary condition file is empty: " + inputs.condPath);
            spdlog::info("Loaded {} fixed vertices from: {}", fixedVertices.size(), inputs.condPath);
        }

        // ================================================================
        // Phase 2: Loop subdivision (mesh + material synced)
        // ================================================================
        //   Original vertices keep their indices (0..nV_old-1),
        //   so fixedVertices indices remain valid after subdivision.
        //   Each parent face i -> children 4*i .. 4*i+3.

        if (inputs.nfMin > 0) {
            while (static_cast<int>(F.rows()) < inputs.nfMin) {
                const int nF_old = static_cast<int>(F.rows());
                const int nV_old = static_cast<int>(P.rows());

                Eigen::MatrixXd V3 = makeFlatFromParam(P);

                // Linear midpoint subdivision: split each triangle into 4
                // without smoothing (preserves parameterization exactly).
                // Edge midpoints are placed at linear average of endpoints.
                std::map<std::pair<int,int>, int> edgeMidpoint;
                int nextVid = nV_old;
                Eigen::MatrixXd P_new(nV_old + 3 * nF_old, 2); // upper bound
                P_new.topRows(nV_old) = P;

                auto getMid = [&](int a, int b) -> int {
                    auto key = std::make_pair(std::min(a,b), std::max(a,b));
                    auto it = edgeMidpoint.find(key);
                    if (it != edgeMidpoint.end()) return it->second;
                    int mid = nextVid++;
                    P_new(mid, 0) = 0.5 * (P(a, 0) + P(b, 0));
                    P_new(mid, 1) = 0.5 * (P(a, 1) + P(b, 1));
                    edgeMidpoint[key] = mid;
                    return mid;
                };

                Eigen::MatrixXi F_sub(4 * nF_old, 3);
                for (int fi = 0; fi < nF_old; ++fi) {
                    int v0 = F(fi, 0), v1 = F(fi, 1), v2 = F(fi, 2);
                    int m01 = getMid(v0, v1);
                    int m12 = getMid(v1, v2);
                    int m20 = getMid(v2, v0);
                    F_sub.row(4*fi + 0) << v0,  m01, m20;
                    F_sub.row(4*fi + 1) << m01, v1,  m12;
                    F_sub.row(4*fi + 2) << m20, m12, v2;
                    F_sub.row(4*fi + 3) << m01, m12, m20;
                }
                P = P_new.topRows(nextVid);
                F_sub.conservativeResize(4 * nF_old, 3);

                // Subdivide material: children inherit parent
                const int nF_new = static_cast<int>(F_sub.rows());
                Eigen::VectorXd t1_sub(nF_new), t2_sub(nF_new);
                for (int fi = 0; fi < nF_old; ++fi) {
                    for (int c = 0; c < 4; ++c) {
                        t1_sub[4 * fi + c] = t1[fi];
                        t2_sub[4 * fi + c] = t2[fi];
                    }
                }
                t1 = std::move(t1_sub);
                t2 = std::move(t2_sub);
                F = F_sub;

                spdlog::info("Loop subdivided -> {} vertices, {} faces.", P.rows(), F.rows());
            }
        }

        const int nF = static_cast<int>(F.rows());
        const int nV = static_cast<int>(P.rows());

        // ================================================================
        // Phase 3: Build simulation structures
        // ================================================================

        const Eigen::MatrixXd V_flat = makeFlatFromParam(P);
        ManifoldSurfaceMesh    mesh(F);
        VertexPositionGeometry geometry(mesh, V_flat);
        geometry.refreshQuantities();

        FaceData<Eigen::Matrix2d> MrInv = precomputeMrInv(mesh, P, F);

        // Diagnostic: check vertex ordering consistency
        {
            int match = 0, rotated = 0, mismatch = 0;
            for (Face f : mesh.faces()) {
                int fi = static_cast<int>(f.getIndex());
                Halfedge he = f.halfedge();
                int v0 = static_cast<int>(he.vertex().getIndex());
                int v1 = static_cast<int>(he.next().vertex().getIndex());
                int v2 = static_cast<int>(he.next().next().vertex().getIndex());
                if (v0==F(fi,0) && v1==F(fi,1) && v2==F(fi,2)) match++;
                else if ((v0==F(fi,1)&&v1==F(fi,2)&&v2==F(fi,0)) ||
                         (v0==F(fi,2)&&v1==F(fi,0)&&v2==F(fi,1))) rotated++;
                else mismatch++;
            }
            spdlog::info("Vertex ordering: {} match, {} rotated, {} mismatch out of {} faces.",
                         match, rotated, mismatch, nF);
            if (rotated + mismatch > 0)
                spdlog::warn("VERTEX ORDERING MISMATCH: MrInv uses F matrix order, "
                             "but energy uses halfedge order. This causes WRONG deformation gradient!");
        }

        // Build boundary-face reference mapping for shape operator computation.
        std::vector<bool> is_boundary(false);
        std::vector<int> ref_faces = buildRefFaces(mesh, is_boundary);

        // -- Build fixed DOF indices (3 DOFs per vertex) ------------------
        std::vector<int> fixedIdx;
        if (!fixedVertices.empty()) {
            for (int vid : fixedVertices) {
                if (vid < 0 || vid >= nV)
                    throw std::runtime_error(
                        "Fixed vertex index " + std::to_string(vid)
                        + " out of range [0, " + std::to_string(nV) + ")");
                fixedIdx.push_back(3 * vid);
                fixedIdx.push_back(3 * vid + 1);
                fixedIdx.push_back(3 * vid + 2);
            }
            std::sort(fixedIdx.begin(), fixedIdx.end());
            fixedIdx.erase(std::unique(fixedIdx.begin(), fixedIdx.end()),
                           fixedIdx.end());
            spdlog::info("Fixed {} DOFs from boundary condition.", fixedIdx.size());
        } else {
            fixedIdx = findCenterFaceIndices(P, F);  // already sorted internally
            spdlog::info("Using center-face fixed DOFs ({} DOFs).", fixedIdx.size());
        }
        if (fixedIdx.empty())
            throw std::runtime_error(
                "Forward requires at least one fixed vertex to remove rigid-body modes.");
        spdlog::info("Forward solve keeps {} constrained DOFs.", fixedIdx.size());

        // -- Convert (t1, t2) -> per-face (lambda, kappa) -----------------
        FaceData<double> lambda_pf(mesh);
        FaceData<double> kappa_pf(mesh);
        for (Face f : mesh.faces()) {
            const int fi = static_cast<int>(f.getIndex());
            lambda_pf[f] = compute_lamb_d(ac.m_strain_curve, t1[fi], t2[fi]);
            kappa_pf[f]  = compute_curv_d(ac.m_strain_curve, ac.thickness,
                                          t1[fi], t2[fi]);
            // printf("Face %d: t1=%.3f, t2=%.3f -> lambda=%.3e, kappa=%.3e\n",
            //        fi, t1[fi], t2[fi], lambda_pf[f], kappa_pf[f]);
        }

        // -- Build simulation energy function -----------------------------
        auto simFunc = simulationFunction(
            geometry, MrInv, lambda_pf, kappa_pf,
            kE, kNu, ac.thickness, kWs, kWb, ref_faces);

        // -- Forward solve from flat initial state ------------------------
        Eigen::MatrixXd Vr = V_flat;
        spdlog::info("Starting Newton forward solve (max_iter={}, eps={}).",
                     inputs.maxIter, inputs.epsilon);

        newton(geometry, Vr, simFunc,
               inputs.maxIter, inputs.epsilon, /*verbose=*/true, fixedIdx);

        // -- Write OBJ output ---------------------------------------------
        ensureParentDir(inputs.outputPath);
        if (!igl::writeOBJ(inputs.outputPath, Vr, F))
            throw std::runtime_error("Could not write output OBJ: "
                                     + inputs.outputPath);
        spdlog::info("Forward equilibrium shape written to: {}", inputs.outputPath);

        // -- Compute per-face diagnostics and write VTK -------------------
        {
            // Actual (lambda, kappa) from deformed shape:
            // ComputeMorphophing needs geometry built from deformed Vr,
            // but uses the same reference MrInv (from flat parameterization).
            const int nVr = static_cast<int>(Vr.rows());
            const int nFr = nF;
            Eigen::VectorXd lam_pv_r = Eigen::VectorXd::Zero(nVr);
            Eigen::VectorXd lam_pf_r = Eigen::VectorXd::Zero(nFr);
            Eigen::VectorXd kap_pv_r = Eigen::VectorXd::Zero(nVr);
            Eigen::VectorXd kap_pf_r = Eigen::VectorXd::Zero(nFr);
            Morphmesh::ComputeMorphophing(geometry, Vr, F, nVr, nFr,
                MrInv,
                lam_pv_r, lam_pf_r, kap_pv_r, kap_pf_r);

            for(Face f : mesh.faces()) {
                
                int fi = static_cast<int>(f.getIndex());
                if(is_boundary[fi])
                    printf("Face %d: lambda_target=%.3e, lambda_actual=%.3e, kappa_target=%.3e, kappa_actual=%.3e\n",
                        fi, lambda_pf[f], lam_pf_r[fi], kappa_pf[f], kap_pf_r[fi]);
            }

            // Target (lambda, kappa) per face
            Eigen::VectorXd lam_pf_t(nFr), kap_pf_t(nFr);
            for (Face f : mesh.faces()) {
                int fi = static_cast<int>(f.getIndex());
                lam_pf_t[fi] = lambda_pf[f];
                kap_pf_t[fi] = kappa_pf[f];
            }

            // Differences
            Eigen::VectorXd lam_diff = lam_pf_r - lam_pf_t;
            Eigen::VectorXd kap_diff = kap_pf_r - kap_pf_t;

            // Energy densities (use original geometry + MrInv, deformed Vr)
            Eigen::VectorXd Ws_density = Eigen::VectorXd::Zero(nFr);
            Eigen::VectorXd Wb_density = Eigen::VectorXd::Zero(nFr);
            Morphmesh morph_diag(Vr, P, F, kE, kNu);
            morph_diag.ComputeElasticEnergy(geometry, MrInv,
                lambda_pf, kappa_pf, ac.thickness,
                Vr, F, Ws_density, Wb_density);

            // Write single VTK with all per-face diagnostics
            std::string vtkPath = inputs.outputPath;
            {
                auto dotPos = vtkPath.rfind(".obj");
                if (dotPos != std::string::npos && dotPos == vtkPath.size() - 4)
                    vtkPath.replace(dotPos, 4, ".vtk");
                else
                    vtkPath += ".vtk";
            }

            std::vector<Eigen::VectorXd> attrs = {
                lam_pf_t, kap_pf_t,         // target (material intrinsic)
                lam_pf_r, kap_pf_r,         // actual (from deformation)
                lam_diff, kap_diff,          // differences
                Ws_density, Wb_density       // energy densities
            };
            std::vector<std::string> names = {
                "lambda_target", "kappa_target",
                "lambda_actual", "kappa_actual",
                "lambda_diff", "kappa_diff",
                "Ws_density", "Wb_density"
            };

            write_output_vtk_perface(vtkPath, Vr, F, attrs, names);
            spdlog::info("VTK diagnostics written to: {}", vtkPath);
        }

        return EXIT_SUCCESS;

    } catch (const std::exception& e) {
        spdlog::error("Forward simulation failed: {}", e.what());
        return EXIT_FAILURE;
    }
}
