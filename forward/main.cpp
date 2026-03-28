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

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

#include <spdlog/spdlog.h>

#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"
#include "functions.h"
#include "material.hpp"
#include "morph_functions.hpp"
#include "newton.h"
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
    int         maxIter = 20;
    double      epsilon = 1e-6;
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

/// Resolve final file paths from either config or explicit options.
ForwardInputs resolveInputs(const CliOptions& opts)
{
    ForwardInputs inputs;

    if (!opts.cfgPath.empty()) {
        const Config cfg(opts.cfgPath);
        if (cfg.model.name.empty())
            throw std::runtime_error("Config model.name is empty.");

        inputs.paramPath    = cfg.paramDir()   + cfg.model.name + "_param.obj";
        inputs.materialPath = cfg.designDir()  + cfg.model.name + "_material.txt";
        inputs.curvesPath   = cfg.material.curves_path;
        inputs.outputPath   = !opts.outputPath.empty()
            ? opts.outputPath
            : cfg.output.output_path + cfg.model.name + "_forward.obj";
        inputs.maxIter = cfg.solver.max_iter;
        inputs.epsilon = cfg.solver.epsilon;
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

        // -- Load parameterized mesh --------------------------------------
        const auto [P, F] = loadParamMesh(inputs.paramPath);
        const int nF = static_cast<int>(F.rows());
        spdlog::info("Loaded param mesh: {} vertices, {} faces.", P.rows(), nF);

        // -- Load per-face material assignments ---------------------------
        auto [t1, t2] = loadMaterial(inputs.materialPath, nF);
        spdlog::info("Loaded material assignments for {} faces.", nF);

        // -- Load material curves -----------------------------------------
        ActiveComposite ac(inputs.curvesPath);
        ac.ComputeMaterialCurve();
        if (ac.thickness <= 0.0)
            throw std::runtime_error(
                "Material thickness must be positive (got "
                + std::to_string(ac.thickness) + ").");
        spdlog::info("Material curves loaded (thickness = {}).", ac.thickness);

        // -- Build geometry-central structures from flat mesh --------------
        const Eigen::MatrixXd V_flat = makeFlatFromParam(P);
        ManifoldSurfaceMesh    mesh(F);
        VertexPositionGeometry geometry(mesh, V_flat);
        geometry.refreshQuantities();

        // -- Precompute per-face inverse reference shape -------------------
        FaceData<Eigen::Matrix2d> MrInv = precomputeMrInv(mesh, P, F);

        // -- Fixed DOFs for rigid-body removal ----------------------------
        const std::vector<int> fixedIdx = findCenterFaceIndices(P, F);
        spdlog::info("Fixed {} DOFs for rigid-body removal.", fixedIdx.size());

        // -- Convert (t1, t2) -> per-face (lambda, kappa) -----------------
        FaceData<double> lambda_pf(mesh);
        FaceData<double> kappa_pf(mesh);
        for (Face f : mesh.faces()) {
            const int fi = static_cast<int>(f.getIndex());
            lambda_pf[f] = compute_lamb_d(ac.m_strain_curve, t1[fi], t2[fi]);
            kappa_pf[f]  = compute_curv_d(ac.m_strain_curve, ac.thickness,
                                          t1[fi], t2[fi]);
        }

        // -- Build simulation energy function -----------------------------
        auto simFunc = simulationFunction(
            geometry, MrInv, lambda_pf, kappa_pf,
            kE, kNu, ac.thickness, kWs, kWb);

        // -- Forward solve from flat initial state ------------------------
        Eigen::MatrixXd Vr = V_flat;
        spdlog::info("Starting Newton forward solve (max_iter={}, eps={}).",
                     inputs.maxIter, inputs.epsilon);

        newton(geometry, Vr, simFunc,
               inputs.maxIter, inputs.epsilon, /*verbose=*/true, fixedIdx);

        // -- Write output -------------------------------------------------
        ensureParentDir(inputs.outputPath);
        if (!igl::writeOBJ(inputs.outputPath, Vr, F))
            throw std::runtime_error("Could not write output OBJ: "
                                     + inputs.outputPath);
        spdlog::info("Forward equilibrium shape written to: {}", inputs.outputPath);

        return EXIT_SUCCESS;

    } catch (const std::exception& e) {
        spdlog::error("Forward simulation failed: {}", e.what());
        return EXIT_FAILURE;
    }
}
