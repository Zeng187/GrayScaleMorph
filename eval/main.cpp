/// Standalone Evaluation entry point (replaces FeasEval + PatchMetrics).
///
/// Two modes:
///   --feas     Feasibility: parameterize, compute target (lambda, kappa),
///              report how much they exceed material bounds.
///   --inverse  Ground-truth: run full inverse design, report shape error.
///
/// Usage:
///   ./Evaluate --cfg <cfg.json> --feas
///   ./Evaluate --cfg <cfg.json> --inverse
///   ./Evaluate --mesh <file.obj> --feas      # standalone, hardcoded defaults

#include <Eigen/Core>

#include <igl/loop.h>
#include <igl/readOBJ.h>

#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"
#include "inverse_design.h"
#include "material.hpp"
#include "morphmesh.hpp"
#include "parameterize_pipeline.h"
#include "patch_utils.h"

// ═══════════════════════════════════════════════════════════════════════════
// Anonymous namespace: types, helpers, core logic
// ═══════════════════════════════════════════════════════════════════════════

namespace {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Hardcoded material defaults (for --mesh mode without config)
// ---------------------------------------------------------------------------

constexpr double kDefaultLambdaMin  = 1.025941;
constexpr double kDefaultLambdaMax  = 1.092507;
constexpr double kDefaultKappaMax   = 0.099849;
constexpr double kDefaultPlateWidth = 40.0;
constexpr double kDefaultThickness  = 1.0;

// ---------------------------------------------------------------------------
// StVK energy constants (nu = 0.5)
// ---------------------------------------------------------------------------

constexpr double kEtaB = 1.0 / 3.0;   // h^2/3 for thickness h=1
constexpr double kCD   = 1.0 / 6.0;   // C_D/C_iso for nu=0.5
constexpr double kEps  = 1e-30;        // numerical guard

// ---------------------------------------------------------------------------
// Mode selection
// ---------------------------------------------------------------------------

enum class EvalMode { None, Feas, Inverse };

// ---------------------------------------------------------------------------
// Parsed CLI arguments
// ---------------------------------------------------------------------------

struct CliArgs
{
    std::string cfgPath;
    std::string meshPath;        // standalone single-mesh mode
    EvalMode    mode    = EvalMode::None;
    bool        compact = false; // compact JSON (no indentation)
};

// ---------------------------------------------------------------------------
// Material bounds — either from ActiveComposite or hardcoded defaults
// ---------------------------------------------------------------------------

struct MaterialBounds
{
    double lambda_min = kDefaultLambdaMin;
    double lambda_max = kDefaultLambdaMax;
    double kappa_max  = kDefaultKappaMax;
    double platewidth = kDefaultPlateWidth;
    double thickness  = kDefaultThickness;
};

// ---------------------------------------------------------------------------
// Per-piece feasibility summary
// ---------------------------------------------------------------------------

struct FeasSummary
{
    double totalEnergyResidual = 0.0;
    double totalStretchExcess  = 0.0;
    double totalBendExcess     = 0.0;
    int    infeasibleFaces     = 0;
    double lambdaLo =  std::numeric_limits<double>::infinity();
    double lambdaHi = -std::numeric_limits<double>::infinity();
    double kappaLo  =  std::numeric_limits<double>::infinity();
    double kappaHi  = -std::numeric_limits<double>::infinity();

    // Inverse-design fields (only populated in --inverse mode)
    bool   hasInverse = false;
    double distInv    = 0.0;
    double distProj   = 0.0;
};

// ---------------------------------------------------------------------------
// Result for a single mesh piece (whole mesh or one patch)
// ---------------------------------------------------------------------------

struct PieceResult
{
    int         patchId  = -1;   // -1 = whole mesh
    std::string meshPath;
    int         nVertices = 0;
    int         nFaces    = 0;
    double      area      = 0.0;
    FeasSummary summary;
};

// ═══════════════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════════════

void printUsage(const char* prog)
{
    std::cerr
        << "Usage:\n"
        << "  " << prog << " --cfg <cfg.json> --feas\n"
        << "  " << prog << " --cfg <cfg.json> --inverse\n"
        << "  " << prog << " --mesh <file.obj> --feas\n"
        << "\nOptions:\n"
        << "  --cfg <cfg.json>   Config file (default: cfg.json in cwd)\n"
        << "  --mesh <file.obj>  Standalone mesh (uses hardcoded material defaults)\n"
        << "  --feas             Feasibility evaluation\n"
        << "  --inverse          Full inverse-design evaluation\n"
        << "  --compact          Compact JSON output (no indentation)\n";
}

bool parseCli(int argc, char* argv[], CliArgs& args)
{
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];

        if (a == "--cfg" && i + 1 < argc) {
            args.cfgPath = argv[++i];
        } else if (a == "--mesh" && i + 1 < argc) {
            args.meshPath = argv[++i];
        } else if (a == "--feas") {
            if (args.mode != EvalMode::None) {
                std::cerr << "Error: specify exactly one of --feas or --inverse.\n";
                return false;
            }
            args.mode = EvalMode::Feas;
        } else if (a == "--inverse") {
            if (args.mode != EvalMode::None) {
                std::cerr << "Error: specify exactly one of --feas or --inverse.\n";
                return false;
            }
            args.mode = EvalMode::Inverse;
        } else if (a == "--compact") {
            args.compact = true;
        } else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        }
    }

    if (args.mode == EvalMode::None) {
        std::cerr << "Error: must specify --feas or --inverse.\n";
        printUsage(argv[0]);
        return false;
    }
    if (!args.meshPath.empty() && !args.cfgPath.empty()) {
        std::cerr << "Error: --mesh and --cfg are mutually exclusive.\n";
        printUsage(argv[0]);
        return false;
    }
    if (!args.meshPath.empty() && args.mode != EvalMode::Feas) {
        std::cerr << "Error: --mesh only supports --feas mode.\n";
        printUsage(argv[0]);
        return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Geometry helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Total 3D surface area of a triangle mesh.
double computeTotalArea(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    double totalArea = 0.0;
    for (int fi = 0; fi < F.rows(); ++fi) {
        const Eigen::Vector3d e1 = V.row(F(fi, 1)) - V.row(F(fi, 0));
        const Eigen::Vector3d e2 = V.row(F(fi, 2)) - V.row(F(fi, 0));
        totalArea += 0.5 * e1.cross(e2).norm();
    }
    return totalArea;
}

/// Per-face 3D areas for an nF-face mesh.
Eigen::VectorXd computePerFaceAreas(const Eigen::MatrixXd& V,
                                     const Eigen::MatrixXi& F)
{
    const int nF = static_cast<int>(F.rows());
    Eigen::VectorXd areas(nF);
    for (int fi = 0; fi < nF; ++fi) {
        const Eigen::Vector3d e1 = V.row(F(fi, 1)) - V.row(F(fi, 0));
        const Eigen::Vector3d e2 = V.row(F(fi, 2)) - V.row(F(fi, 0));
        areas[fi] = 0.5 * e1.cross(e2).norm();
    }
    return areas;
}

/// Scale V in-place so max bounding-box extent equals platewidth. Returns scale.
double scaleToPlatewidth(Eigen::MatrixXd& V, double platewidth)
{
    const double extent =
        (V.colwise().maxCoeff() - V.colwise().minCoeff()).maxCoeff();
    if (extent <= 0.0) return 1.0;
    const double s = platewidth / extent;
    V *= s;
    return s;
}

/// Load an OBJ mesh, throwing on failure.
void loadMesh(const std::string& path,
              Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    if (!igl::readOBJ(path, V, F))
        throw std::runtime_error("Cannot read mesh: " + path);
    if (V.rows() == 0 || F.rows() == 0)
        throw std::runtime_error("Mesh is empty: " + path);
}

/// Loop-subdivide until face count reaches minimum.
void subdivideToMin(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int nfMin)
{
    while (static_cast<int>(F.rows()) < nfMin) {
        Eigen::MatrixXd Vtmp = V;
        Eigen::MatrixXi Ftmp = F;
        igl::loop(Vtmp, Ftmp, V, F);
    }
}

/// Find seg_id.txt path from config (primary + fallback). Returns "" if not found.
std::string findSegIdPath(const Config& cfg)
{
    const std::string primary = cfg.segmentDir(cfg.model.name) + "seg_id.txt";
    if (std::filesystem::exists(primary))
        return primary;

    if (!cfg.segment.method.empty()) {
        const std::string fallback =
            cfg.segment.path + cfg.model.name + "/seg_id.txt";
        if (std::filesystem::exists(fallback))
            return fallback;
    }
    return "";
}

/// Build MaterialBounds from a loaded ActiveComposite + config.
MaterialBounds boundsFromMaterial(const ActiveComposite& ac, const Config& cfg)
{
    MaterialBounds b;
    b.lambda_min = ac.range_lam.x;
    b.lambda_max = ac.range_lam.y;
    b.kappa_max  = std::max(std::abs(ac.range_kap.x),
                            std::abs(ac.range_kap.y));
    b.platewidth = cfg.solver.platewidth;
    b.thickness  = ac.thickness;
    return b;
}

// ═══════════════════════════════════════════════════════════════════════════
// Core feasibility evaluation for a single mesh piece
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate feasibility (and optionally inverse design) on a single mesh piece.
///
/// The caller must have already scaled V to platewidth.  For --inverse mode,
/// `solver` and `ac` must be non-null.
PieceResult evaluatePiece(
    const Eigen::MatrixXd& V_scaled,
    const Eigen::MatrixXi& F_in,
    const std::string&     meshPath,
    const MaterialBounds&  bounds,
    EvalMode               mode,
    const Config::Solver*  solver,   // null in standalone --mesh mode
    const ActiveComposite* ac,       // null in standalone --mesh mode
    int                    patchId)
{
    PieceResult piece;
    piece.patchId   = patchId;
    piece.meshPath  = meshPath;
    piece.nVertices = static_cast<int>(V_scaled.rows());
    piece.nFaces    = static_cast<int>(F_in.rows());
    piece.area      = computeTotalArea(V_scaled, F_in);

    // -- Parameterization --------------------------------------------------
    ParameterizeResult param = parameterizeMesh(
        V_scaled, F_in,
        bounds.lambda_min, bounds.lambda_max, bounds.platewidth);

    if (!param.mesh || !param.geometry)
        throw std::runtime_error("parameterizeMesh returned null mesh/geometry.");

    const int nV = static_cast<int>(param.V.rows());
    const int nF = static_cast<int>(param.F.rows());

    // -- Compute target (lambda, kappa) ------------------------------------
    Eigen::VectorXd lam_pv = Eigen::VectorXd::Zero(nV);
    Eigen::VectorXd lam_pf = Eigen::VectorXd::Zero(nF);
    Eigen::VectorXd kap_pv = Eigen::VectorXd::Zero(nV);
    Eigen::VectorXd kap_pf = Eigen::VectorXd::Zero(nF);

    Morphmesh::ComputeMorphophing(
        *param.geometry, param.V, param.F,
        nV, nF, param.MrInv,
        lam_pv, lam_pf, kap_pv, kap_pf);

    // -- Compute per-face Gaussian curvature (det of shape operator) -------
    // We replicate the shape-operator computation from ComputeMorphophing to
    // extract K = det(a^{-1} * b) alongside H = 0.5 * tr(a^{-1} * b).
    Eigen::VectorXd gauss_pf = Eigen::VectorXd::Zero(nF);
    {
        using namespace geometrycentral::surface;
        SurfaceMesh& mesh = param.geometry->mesh;

        for (int fi = 0; fi < nF; ++fi) {
            Face f = mesh.face(fi);

            const int i0 = f.halfedge().vertex().getIndex();
            const int i1 = f.halfedge().next().vertex().getIndex();
            const int i2 = f.halfedge().next().next().vertex().getIndex();

            Eigen::Matrix<double, 3, 2> M;
            M.col(0) = param.V.row(i1) - param.V.row(i0);
            M.col(1) = param.V.row(i2) - param.V.row(i0);

            const Eigen::Matrix<double, 3, 2> Fg = M * param.MrInv[f];

            // Face normal (un-normalised)
            const Eigen::Vector3d n = M.col(0).cross(M.col(1));
            const double n2 = n.squaredNorm();
            if (n2 <= kEps) continue;

            // Discrete shape operator L via dihedral angles
            Eigen::Matrix3d L = Eigen::Matrix3d::Zero();
            for (Halfedge he : f.adjacentHalfedges()) {
                if (he.edge().isBoundary()) continue;

                const int v0 = he.vertex().getIndex();
                const int v1 = he.next().vertex().getIndex();
                const int v2 = he.twin().next().next().vertex().getIndex();

                const Eigen::Vector3d e1 = param.V.row(v1) - param.V.row(v0);
                const Eigen::Vector3d e2 = param.V.row(v2) - param.V.row(v0);

                const Eigen::Vector3d nAdj = e2.cross(e1);
                const double theta = std::atan2(
                    n.cross(nAdj).dot(e1),
                    e1.norm() * nAdj.dot(n));

                const Eigen::Vector3d t = n.cross(e1);
                const double tNorm = t.norm();
                if (tNorm <= kEps) continue;

                L += theta * (t / tNorm) * t.transpose();
            }
            L /= n2;

            const Eigen::Matrix2d a_mat = Fg.transpose() * Fg;
            const Eigen::Matrix2d b_mat = Fg.transpose() * L * Fg;
            const double detA = a_mat.determinant();
            if (std::abs(detA) <= kEps) continue;

            const Eigen::Matrix2d S = a_mat.inverse() * b_mat;
            gauss_pf[fi] = S.determinant();
        }
    }

    // -- Per-face 3D areas -------------------------------------------------
    const Eigen::VectorXd area3d = computePerFaceAreas(param.V, param.F);

    // -- Gauge shift: find t* minimizing area-weighted StVK residual -------
    // This matches the old FeasEval golden-section search exactly.

    auto psiAtFace = [&](int fi, double t) -> double {
        const double s     = t * lam_pf[fi];
        const double H     = kap_pf[fi];
        const double K     = gauss_pf[fi];
        const double Q     = std::max(0.0, H * H - K);

        const double s2    = s * s;
        const double lamS  = std::clamp(s, bounds.lambda_min, bounds.lambda_max);
        const double lam2  = lamS * lamS;

        // Stretch: StVK quartic
        const double psiS  = (s2 - lam2) * (s2 - lam2)
                             / std::max(lam2, kEps);

        // Bending: coupled via r = s^2 / lambda*^2
        const double r     = s2 / std::max(lam2, kEps);
        const double Heff  = r * H;
        const double kapS  = std::clamp(Heff, -bounds.kappa_max,
                                                bounds.kappa_max);
        const double bendIso   = (Heff - kapS) * (Heff - kapS);
        const double bendAniso = 2.0 * kCD * r * r * Q;

        return psiS + kEtaB * lam2 * (bendIso + bendAniso);
    };

    // Golden-section search over t in [t_lo, t_hi]
    {
        const double lamMinRaw = lam_pf.head(nF).minCoeff();
        const double lamMaxRaw = lam_pf.head(nF).maxCoeff();

        double tLo = 0.1, tHi = 10.0;
        if (lamMaxRaw > 1e-15)
            tLo = std::max(0.1, bounds.lambda_min / (lamMaxRaw + 1e-15) * 0.5);
        if (lamMinRaw > 1e-15)
            tHi = std::min(10.0, bounds.lambda_max / (lamMinRaw + 1e-15) * 2.0);
        if (tLo >= tHi) { tLo = 0.1; tHi = 10.0; }

        auto totalResidual = [&](double t) -> double {
            double sum = 0.0;
            for (int fi = 0; fi < nF; ++fi)
                sum += area3d[fi] * psiAtFace(fi, t);
            return sum;
        };

        const double phi = (std::sqrt(5.0) - 1.0) / 2.0;
        double a = tLo, b = tHi;
        double c = b - phi * (b - a);
        double d = a + phi * (b - a);

        for (int iter = 0; iter < 50; ++iter) {
            if (totalResidual(c) < totalResidual(d))
                b = d;
            else
                a = c;
            c = b - phi * (b - a);
            d = a + phi * (b - a);
        }

        const double tStar = 0.5 * (a + b);

        // Apply gauge shift to lambda (kappa is unchanged)
        lam_pf *= tStar;
        lam_pv *= tStar;

        spdlog::info("Piece {}: gauge shift t*={:.6f}, lambda range after=[{:.6f}, {:.6f}]",
                     patchId, tStar,
                     lam_pf.head(nF).minCoeff(),
                     lam_pf.head(nF).maxCoeff());
    }

    // -- Feasibility metrics (StVK-aligned, after gauge shift) -------------

    FeasSummary& s = piece.summary;

    for (int fi = 0; fi < nF; ++fi) {
        const double lam  = lam_pf[fi];
        const double H    = kap_pf[fi];
        const double K    = gauss_pf[fi];
        const double Q    = std::max(0.0, H * H - K);

        s.lambdaLo = std::min(s.lambdaLo, lam);
        s.lambdaHi = std::max(s.lambdaHi, lam);
        s.kappaLo  = std::min(s.kappaLo, H);
        s.kappaHi  = std::max(s.kappaHi, H);

        const double s2   = lam * lam;
        const double lamS = std::clamp(lam, bounds.lambda_min, bounds.lambda_max);
        const double lam2 = lamS * lamS;
        const double r    = s2 / std::max(lam2, kEps);
        const double Heff = r * H;

        // Excess beyond material bounds
        const double stretchEx =
            std::max(0.0, lam - bounds.lambda_max)
            + std::max(0.0, bounds.lambda_min - lam);
        const double bendEx =
            std::max(0.0, std::abs(Heff) - bounds.kappa_max);

        // StVK energy residual
        const double kapS     = std::clamp(Heff, -bounds.kappa_max,
                                                   bounds.kappa_max);
        const double psiS     = (s2 - lam2) * (s2 - lam2) / std::max(lam2, kEps);
        const double bendIso  = (Heff - kapS) * (Heff - kapS);
        const double bendAni  = 2.0 * kCD * r * r * Q;
        const double energyR  = area3d[fi]
                                * (psiS + kEtaB * lam2 * (bendIso + bendAni));

        s.totalStretchExcess  += stretchEx;
        s.totalBendExcess     += bendEx;
        s.totalEnergyResidual += energyR;
        if (stretchEx > 0.0 || bendEx > 0.0)
            ++s.infeasibleFaces;
    }

    // -- Inverse design (if requested) -------------------------------------

    if (mode == EvalMode::Inverse) {
        if (!solver || !ac)
            throw std::runtime_error(
                "Inverse mode requires solver settings and material.");

        InverseDesignProblem prob;
        prob.V         = param.V;
        prob.F         = param.F;
        prob.P         = param.P;
        prob.mesh      = param.mesh.get();
        prob.geometry  = param.geometry.get();
        prob.MrInv     = param.MrInv;
        prob.fixedIdx  = param.fixedIdx;
        prob.ac        = ac;

        prob.max_iter          = solver->max_iter;
        prob.epsilon           = solver->epsilon;
        prob.w_s               = solver->w_s;
        prob.w_b               = solver->w_b;
        prob.wM_kap            = solver->wM_kap;
        prob.wL_kap            = solver->wL_kap;
        prob.wM_lam            = solver->wM_lam;
        prob.wL_lam            = solver->wL_lam;
        prob.wP_kap            = solver->wP_kap;
        prob.wP_lam            = solver->wP_lam;
        prob.penalty_threshold = solver->penalty_threshold;
        prob.betaP             = solver->betaP;
        prob.patch_id          = patchId;

        const InverseDesignResult inv = runInverseDesign(prob);

        s.hasInverse = true;
        s.distInv    = inv.dist_inv;
        s.distProj   = inv.dist_proj;
    }

    return piece;
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON serialisation (backward-compatible with FeasEval output)
// ═══════════════════════════════════════════════════════════════════════════

/// Sanitise range bounds (replace +/-inf with 0).
std::pair<double, double> sanitiseRange(double lo, double hi)
{
    if (!std::isfinite(lo) || !std::isfinite(hi))
        return {0.0, 0.0};
    return {lo, hi};
}

json materialToJson(const MaterialBounds& b)
{
    return {
        {"lambda_min", b.lambda_min},
        {"lambda_max", b.lambda_max},
        {"kappa_max",  b.kappa_max},
        {"thickness",  b.thickness},
    };
}

json summaryToJson(const FeasSummary& s)
{
    auto [lamLo, lamHi] = sanitiseRange(s.lambdaLo, s.lambdaHi);
    auto [kapLo, kapHi] = sanitiseRange(s.kappaLo, s.kappaHi);

    json j = {
        {"total_energy_residual", s.totalEnergyResidual},
        {"total_stretch_excess",  s.totalStretchExcess},
        {"total_bend_excess",     s.totalBendExcess},
        {"infeasible_faces",      s.infeasibleFaces},
        {"lambda_range",          {lamLo, lamHi}},
        {"kappa_range",           {kapLo, kapHi}},
    };

    if (s.hasInverse) {
        j["dist_inv"]  = s.distInv;
        j["dist_proj"] = s.distProj;
    }

    return j;
}

json patchResultToJson(const PieceResult& p, const MaterialBounds& b)
{
    json j = {
        {"patch_id", p.patchId},
        {"mesh", {
            {"path",     p.meshPath},
            {"vertices", p.nVertices},
            {"faces",    p.nFaces},
            {"area",     p.area},
        }},
        {"summary", summaryToJson(p.summary)},
    };
    return j;
}

/// Aggregate multiple patch summaries into a global summary.
FeasSummary aggregateSummaries(const std::vector<PieceResult>& patches)
{
    FeasSummary agg;
    double weightSum = 0.0;
    double distInvSum = 0.0;
    double distProjSum = 0.0;

    for (const auto& p : patches) {
        agg.totalEnergyResidual += p.summary.totalEnergyResidual;
        agg.totalStretchExcess  += p.summary.totalStretchExcess;
        agg.totalBendExcess     += p.summary.totalBendExcess;
        agg.infeasibleFaces     += p.summary.infeasibleFaces;
        agg.lambdaLo = std::min(agg.lambdaLo, p.summary.lambdaLo);
        agg.lambdaHi = std::max(agg.lambdaHi, p.summary.lambdaHi);
        agg.kappaLo  = std::min(agg.kappaLo,  p.summary.kappaLo);
        agg.kappaHi  = std::max(agg.kappaHi,  p.summary.kappaHi);

        if (p.summary.hasInverse) {
            agg.hasInverse = true;
            // Weight by vertex count: dist_inv/dist_proj are per-vertex MSEs
            const double w = static_cast<double>(p.nVertices);
            distInvSum  += p.summary.distInv  * w;
            distProjSum += p.summary.distProj * w;
            weightSum   += w;
        }
    }

    if (agg.hasInverse && weightSum > 0.0) {
        agg.distInv  = distInvSum  / weightSum;
        agg.distProj = distProjSum / weightSum;
    }

    return agg;
}

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Entry point
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[])
{
    try {
        // Suppress spdlog output — all structured output goes to stdout as JSON,
        // diagnostic info goes to stderr via spdlog.
        spdlog::set_level(spdlog::level::warn);

        // -- Parse CLI -----------------------------------------------------
        CliArgs args;
        if (!parseCli(argc, argv, args))
            return EXIT_FAILURE;

        json output;

        // ==================================================================
        // Standalone --mesh mode (no config, hardcoded defaults)
        // ==================================================================

        if (!args.meshPath.empty()) {
            Eigen::MatrixXd V;
            Eigen::MatrixXi F;
            loadMesh(args.meshPath, V, F);

            // Match old FeasEval: scale by bounding-box diagonal
            {
                const Eigen::RowVector3d lo = V.colwise().minCoeff();
                const Eigen::RowVector3d hi = V.colwise().maxCoeff();
                const double diag = (hi - lo).norm();
                if (diag > 0.0) V *= kDefaultPlateWidth / diag;
            }

            MaterialBounds bounds; // uses hardcoded defaults

            const PieceResult piece = evaluatePiece(
                V, F, args.meshPath, bounds,
                EvalMode::Feas,
                /*solver=*/nullptr, /*ac=*/nullptr,
                /*patchId=*/-1);

            output = {
                {"mesh", {
                    {"path",     piece.meshPath},
                    {"vertices", piece.nVertices},
                    {"faces",    piece.nFaces},
                    {"area",     piece.area},
                }},
                {"material", materialToJson(bounds)},
                {"summary",  summaryToJson(piece.summary)},
            };

            std::cout << output.dump(args.compact ? -1 : 2) << "\n";
            return EXIT_SUCCESS;
        }

        // ==================================================================
        // Config mode
        // ==================================================================

        const std::string cfgPath =
            args.cfgPath.empty() ? "cfg.json" : args.cfgPath;
        Config config(cfgPath);

        ActiveComposite ac(config.material.curves_path);
        ac.ComputeMaterialCurve();
        ac.ComputeFeasibleVals();

        const MaterialBounds bounds = boundsFromMaterial(ac, config);

        // Load and optionally subdivide
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        loadMesh(config.model.mesh_path, V, F);
        subdivideToMin(V, F, config.solver.nf_min);

        // Check for segmentation
        const std::string segIdPath = findSegIdPath(config);

        if (segIdPath.empty()) {
            // -- Whole-mesh mode -------------------------------------------
            Eigen::MatrixXd Vs = V;
            scaleToPlatewidth(Vs, bounds.platewidth);

            const PieceResult piece = evaluatePiece(
                Vs, F, config.model.mesh_path, bounds,
                args.mode, &config.solver, &ac, /*patchId=*/-1);

            output = {
                {"mesh", {
                    {"path",     piece.meshPath},
                    {"vertices", piece.nVertices},
                    {"faces",    piece.nFaces},
                    {"area",     piece.area},
                }},
                {"material", materialToJson(bounds)},
                {"summary",  summaryToJson(piece.summary)},
            };
        } else {
            // -- Per-patch mode --------------------------------------------
            const std::vector<int> segId = loadSegId(segIdPath);
            if (segId.empty())
                throw std::runtime_error(
                    "Segmentation file is empty: " + segIdPath);
            if (static_cast<int>(segId.size()) != static_cast<int>(F.rows()))
                throw std::runtime_error(
                    "seg_id size (" + std::to_string(segId.size())
                    + ") != face count (" + std::to_string(F.rows()) + ")");

            Eigen::MatrixXd Vs = V;
            scaleToPlatewidth(Vs, bounds.platewidth);

            const int numPatches =
                *std::max_element(segId.begin(), segId.end()) + 1;

            std::vector<PieceResult> patchResults;
            json patchesJson = json::array();

            for (int pid = 0; pid < numPatches; ++pid) {
                PatchData patch = extractPatch(Vs, F, segId, pid);
                if (patch.F.rows() == 0) {
                    spdlog::warn("Patch {} has 0 faces, skipping.", pid);
                    continue;
                }

                try {
                    PieceResult piece = evaluatePiece(
                        patch.V, patch.F,
                        segIdPath + ":patch_" + std::to_string(pid),
                        bounds, args.mode, &config.solver, &ac, pid);

                    patchesJson.push_back(patchResultToJson(piece, bounds));
                    patchResults.push_back(std::move(piece));
                } catch (const std::exception& e) {
                    spdlog::error("Patch {} failed: {}", pid, e.what());
                    // Record a minimal error entry so the JSON shows which
                    // patches were skipped and why.
                    json errEntry = {
                        {"patch_id", pid},
                        {"error",    e.what()},
                    };
                    patchesJson.push_back(std::move(errEntry));
                }
            }

            if (patchResults.empty())
                throw std::runtime_error(
                    "Segmentation produced no non-empty patches.");

            const FeasSummary agg = aggregateSummaries(patchResults);

            output = {
                {"mesh", {
                    {"path",     config.model.mesh_path},
                    {"vertices", static_cast<int>(Vs.rows())},
                    {"faces",    static_cast<int>(F.rows())},
                    {"area",     computeTotalArea(Vs, F)},
                }},
                {"material", materialToJson(bounds)},
                {"summary",  summaryToJson(agg)},
                {"patches",  patchesJson},
            };
        }

        std::cout << output.dump(args.compact ? -1 : 2) << "\n";
        return EXIT_SUCCESS;

    } catch (const std::exception& e) {
        std::cerr << "Evaluate error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
