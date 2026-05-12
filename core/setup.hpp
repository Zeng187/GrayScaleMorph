#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

/// Global physical and fabrication parameters loaded from
/// `Resources/setup/global.json`. These are shared across all modules
/// (GrayScaleMorph, ProxyEval, and the Python toolchain) and should be the
/// single source of truth for any of the four scalars below.
///
/// Per-experiment cfg.json files may override individual fields by repeating
/// the key inside the cfg; the loader logs a warning when this happens so
/// that the override stays auditable.
struct GlobalSetup
{
    std::string schema_version = "1.0.0";
    double platewidth    = 40.0;   // mm — target bbox extent (geometric semantic varies by module; see Resources/setup/MODULE.md)
    double thickness     = 1.0;    // mm — physical plate thickness h (enters κ = 1.5·Δε/h)
    double slice_height  = 0.05;   // mm — g-DLP slice z-resolution
    double poisson_ratio = 0.5;    // — incompressible elastomer family

    /// Absolute path of the global.json that produced this instance, kept for
    /// log/error messages so failures can pinpoint which file was loaded.
    std::filesystem::path source_path;
};

struct LoadGlobalSetupOptions
{
    /// Path to the cfg.json that triggered the load. Used both as the search
    /// origin for `Resources/setup/global.json` and as the base directory for
    /// resolving relative `material_path`.
    std::filesystem::path cfg_path;

    /// Optional path to a material file (`{name}.json` or `{name}-poly-curves.json`).
    /// When provided the loader enforces `setup.thickness == material.thickness`
    /// and throws otherwise.
    std::optional<std::filesystem::path> material_path;
};

/// Locate `Resources/setup/global.json`, either via an explicit `setup_path`
/// (or `setup.path`) field in the cfg, or by walking up from the cfg's
/// directory until a `Resources/setup/global.json` is found.
///
/// Throws if no candidate exists.
std::filesystem::path resolveGlobalSetupPath(const nlohmann::json& cfg_root,
                                             const std::filesystem::path& cfg_path);

/// Validate the four scalars are within physically meaningful ranges.
/// Throws on violation.
void validateGlobalSetup(const GlobalSetup& setup);

/// Read the material JSON pointed at by `material_path` (resolved relative to
/// `cfg_path`) and confirm its `thickness` matches `setup.thickness` exactly.
/// Throws on mismatch or missing field.
void validateThicknessAgainstMaterial(const GlobalSetup& setup,
                                      const std::filesystem::path& material_path,
                                      const std::filesystem::path& cfg_path);

/// Top-level entry point: locate global.json, parse + validate it, optionally
/// cross-check against a material file. Logs the loaded values via spdlog.
GlobalSetup loadGlobalSetup(const nlohmann::json& cfg_root,
                            const LoadGlobalSetupOptions& options);
