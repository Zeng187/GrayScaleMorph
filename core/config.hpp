#pragma once

#include <string>

/// Central configuration for the GrayScaleMorph pipeline.
///
/// Reads a JSON file organised into five sections:
///   model, material, segment, output, solver.
///
/// For backward compatibility the parser also accepts the legacy
/// section/key names and the old array-wrapping convention (e.g. ["value"]).
class Config
{
public:
    explicit Config(const std::string& filePath);

    // ── Model ────────────────────────────────────────────────────────
    struct Model {
        std::string name;       ///< e.g. "hemisphere"
        std::string mesh_path;  ///< full relative path to .obj
    } model;

    // ── Material ─────────────────────────────────────────────────────
    struct Material {
        std::string curves_path; ///< path to poly-curves JSON
    } material;

    // ── Segmentation ─────────────────────────────────────────────────
    struct Segment {
        std::string path;    ///< base directory for segment outputs
        std::string method;  ///< distortion method tag (may be empty)
        std::string plan;    ///< plan tag (may be empty)
    } segment;

    // ── Shared resource paths (read/write by multiple modules) ──────
    struct Paths {
        std::string param_path;   ///< parameterization results (Parameterize writes, Forward/Inverse read)
        std::string morph_path;   ///< inverse design intermediates (Inverse writes)
        std::string design_path;  ///< material design files (Inverse writes, Forward/Simulate read)
        std::string output_path;  ///< local output directory
    } paths;

    // ── Optional single-patch override ───────────────────────────────
    struct Patch {
        int id = -1;  ///< patch id to process; < 0 means process all patches
    } patch;

    // ── Solver parameters ────────────────────────────────────────────
    struct Solver {
        double platewidth       = 40.0;
        int    max_iter         = 20;
        int    nf_min           = 200;
        double epsilon          = 1e-6;
        double w_s              = 1.0;
        double w_b              = 1.0;
        double wM_kap           = 0.1;
        double wL_kap           = 0.1;
        double wM_lam           = 0.0;
        double wL_lam           = 0.1;
        double wP_kap           = 0.01;
        double wP_lam           = 0.01;
        double penalty_threshold = 0.01;
        double betaP            = 50.0;
    } solver;

    // ── Convenience path builders ────────────────────────────────────

    /// Build segment directory: segment.path / {modelName}[_{method}][_{plan}] /
    std::string segmentDir(const std::string& modelName) const;

    /// Build param directory: paths.param_path / model.name /
    std::string paramDir() const;

    /// Build morph directory: paths.morph_path / model.name /
    std::string morphDir() const;

    /// Build design directory: paths.design_path / model.name /
    std::string designDir() const;
};
