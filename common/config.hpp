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

    // ── Output paths ─────────────────────────────────────────────────
    struct Output {
        std::string output_path;  ///< local output directory
        std::string param_path;   ///< parameterization results directory
        std::string morph_path;   ///< shared morph resource directory
        std::string design_path;  ///< shared design resource directory
    } output;

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

    /// Build param output directory: output.param_path / model.name /
    std::string paramDir() const;

    /// Build morph output directory: output.morph_path / model.name /
    std::string morphDir() const;

    /// Build design output directory: output.design_path / model.name /
    std::string designDir() const;
};
