#include "config.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════
namespace {

/// Retrieve a value from a JSON object.
/// Accepts both scalar (`"value"`) and legacy array-wrapped (`["value"]`) forms.
template <typename T>
T jsonGet(const json& obj, const char* key)
{
    const auto& v = obj.at(key);
    if (v.is_array()) {
        if (v.empty())
            throw std::runtime_error(std::string("Config key '") + key + "' is an empty array");
        return v.at(0).get<T>();
    }
    return v.get<T>();
}

/// Like jsonGet but returns `fallback` when the key is absent.
template <typename T>
T jsonGetOr(const json& obj, const char* key, const T& fallback)
{
    if (!obj.contains(key)) return fallback;
    return jsonGet<T>(obj, key);
}

/// Return the first section object found under any of the candidate keys.
/// Throws if none of the keys exist.
const json& findSection(const json& root, std::initializer_list<const char*> keys)
{
    for (const char* k : keys) {
        if (root.contains(k) && root.at(k).is_object())
            return root.at(k);
    }
    std::string msg = "Config: none of the expected sections found (";
    bool first = true;
    for (const char* k : keys) {
        if (!first) msg += ", ";
        msg += k;
        first = false;
    }
    msg += ")";
    throw std::runtime_error(msg);
}

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════
// Construction
// ═══════════════════════════════════════════════════════════════════════

Config::Config(const std::string& filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open()) {
        spdlog::error("Config file not found: {}", filePath);
        std::exit(1);
    }

    json j;
    file >> j;
    spdlog::info("Loaded config from: {}", filePath);

    // ── model ────────────────────────────────────────────────────────
    {
        const auto& sec = findSection(j, {"model", "Model"});
        model.name = jsonGetOr<std::string>(sec, "name", "");
        if (model.name.empty())
            model.name = jsonGet<std::string>(sec, "ModelName");        // legacy

        if (sec.contains("mesh_path")) {
            model.mesh_path = jsonGet<std::string>(sec, "mesh_path");
        } else {
            // Legacy: compose from InputPath + ModelName + Postfix
            auto inputPath  = jsonGet<std::string>(sec, "InputPath");
            auto meshSuffix = jsonGet<std::string>(sec, "Postfix");
            model.mesh_path = inputPath + model.name + meshSuffix;
        }

    }

    // ── material ─────────────────────────────────────────────────────
    {
        const auto& sec = findSection(j, {"material", "Resource"});
        if (sec.contains("curves_path"))
            material.curves_path = jsonGet<std::string>(sec, "curves_path");
        else
            material.curves_path = jsonGet<std::string>(sec, "MaterialPath"); // legacy
    }

    // ── segment ──────────────────────────────────────────────────────
    {
        const auto& sec = findSection(j, {"segment", "Resource"});
        segment.enabled = jsonGetOr<bool>(sec, "enabled", true);

        if (sec.contains("path"))
            segment.path = jsonGet<std::string>(sec, "path");
        else
            segment.path = jsonGetOr<std::string>(sec, "SegmentPath", ""); // legacy; empty OK when disabled

        segment.method = jsonGetOr<std::string>(sec, "method", "");
        if (segment.method.empty())
            segment.method = jsonGetOr<std::string>(sec, "DistortionMethod", "");

        segment.plan = jsonGetOr<std::string>(sec, "plan", "");
        if (segment.plan.empty())
            segment.plan = jsonGetOr<std::string>(sec, "Plan", "");

        if (segment.enabled && segment.path.empty())
            throw std::runtime_error("segment.path is required when segment.enabled=true");
    }

    // ── paths (shared resource directories) ─────────────────────────
    {
        // New format: "paths" section. Legacy fallback: "output" / "OutputSettings".
        const auto& sec = findSection(j, {"paths", "output", "OutputSettings"});

        paths.param_path = jsonGetOr<std::string>(sec, "param_path", "../Resources/param/");

        if (sec.contains("morph_path"))
            paths.morph_path = jsonGet<std::string>(sec, "morph_path");
        else
            paths.morph_path = jsonGetOr<std::string>(sec, "MorphPath", "../Resources/morph/");

        if (sec.contains("design_path"))
            paths.design_path = jsonGet<std::string>(sec, "design_path");
        else
            paths.design_path = jsonGetOr<std::string>(sec, "DesignPath", "../Resources/design/");

        if (sec.contains("cond_path"))
            paths.cond_path = jsonGet<std::string>(sec, "cond_path");
        else
            paths.cond_path = jsonGetOr<std::string>(sec, "CondPath", "../Resources/cond/");
    }

    // ── patch (optional single-patch override) ──────────────────────
    if (j.contains("patch") && j.at("patch").is_object()) {
        const auto& sec = j.at("patch");
        patch.id = jsonGetOr<int>(sec, "id", patch.id);
    }

    // ── global setup (Resources/setup/global.json) ──────────────────
    // Loaded BEFORE solver so that solver fields can default to setup values
    // and only be overridden when cfg explicitly repeats the key.
    {
        LoadGlobalSetupOptions opts;
        opts.cfg_path = std::filesystem::path(filePath);
        if (!material.curves_path.empty())
            opts.material_path = std::filesystem::path(material.curves_path);
        setup = loadGlobalSetup(j, opts);

        solver.platewidth    = setup.platewidth;
        solver.poisson_ratio = setup.poisson_ratio;
    }

    // ── solver ───────────────────────────────────────────────────────
    {
        const auto& sec = findSection(j, {"solver", "RuntimeSettings"});
        // platewidth / poisson_ratio: cfg may override the setup default.
        // We log a warning so any deviation from the global default stays visible.
        if (sec.contains("platewidth") || sec.contains("Platewidth")) {
            const double overridden =
                jsonGetOr(sec, "platewidth", jsonGetOr(sec, "Platewidth", solver.platewidth));
            spdlog::warn("cfg.solver.platewidth={} overrides setup.platewidth={}",
                         overridden, setup.platewidth);
            solver.platewidth = overridden;
        }
        if (sec.contains("poisson_ratio") || sec.contains("PoissonRatio")) {
            const double overridden =
                jsonGetOr(sec, "poisson_ratio", jsonGetOr(sec, "PoissonRatio", solver.poisson_ratio));
            spdlog::warn("cfg.solver.poisson_ratio={} overrides setup.poisson_ratio={}",
                         overridden, setup.poisson_ratio);
            solver.poisson_ratio = overridden;
        }
        solver.max_iter          = jsonGetOr(sec, "max_iter",          jsonGetOr(sec, "MaxIter",    solver.max_iter));
        solver.nf_min            = jsonGetOr(sec, "nf_min",            jsonGetOr(sec, "nFmin",      solver.nf_min));
        solver.epsilon           = jsonGetOr(sec, "epsilon",           solver.epsilon);
        solver.w_s               = jsonGetOr(sec, "w_s",              solver.w_s);
        solver.w_b               = jsonGetOr(sec, "w_b",              solver.w_b);
        solver.wM_kap            = jsonGetOr(sec, "wM_kap",           solver.wM_kap);
        solver.wL_kap            = jsonGetOr(sec, "wL_kap",           solver.wL_kap);
        solver.wM_lam            = jsonGetOr(sec, "wM_lam",           solver.wM_lam);
        solver.wL_lam            = jsonGetOr(sec, "wL_lam",           solver.wL_lam);
        solver.wP_kap            = jsonGetOr(sec, "wP_kap",           solver.wP_kap);
        solver.wP_lam            = jsonGetOr(sec, "wP_lam",           solver.wP_lam);
        solver.penalty_threshold = jsonGetOr(sec, "penalty_threshold", solver.penalty_threshold);
        // Accept the new `well_scale` key (energy-weighted Lorentzian sharpness)
        // or the legacy `betaP` key for backward compatibility.
        if (sec.contains("well_scale"))
            solver.well_scale    = jsonGetOr(sec, "well_scale",       solver.well_scale);
        else if (sec.contains("betaP"))
            solver.well_scale    = jsonGetOr(sec, "betaP",            solver.well_scale);
    }

    spdlog::info("Config: model='{}', modelDir='{}', mesh='{}', segment.enabled={}, patch.id={}",
                 model.name, modelDir(), model.mesh_path, segment.enabled, patch.id);
}

// ═══════════════════════════════════════════════════════════════════════
// Path builders
// ═══════════════════════════════════════════════════════════════════════

std::string Config::modelDir() const
{
    std::string dir = model.name;
    if (!segment.enabled)
        return dir;
    if (!segment.method.empty())
        dir += "_" + segment.method;
    if (!segment.plan.empty())
        dir += "_" + segment.plan;
    return dir;
}

std::string Config::segmentDir() const
{
    return segment.path + modelDir() + "/";
}

std::string Config::paramDir() const
{
    return paths.param_path + modelDir() + "/";
}

std::string Config::morphDir() const
{
    return paths.morph_path + modelDir() + "/";
}

std::string Config::designDir() const
{
    return paths.design_path + modelDir() + "/";
}

std::string Config::condDir() const
{
    return paths.cond_path + modelDir() + "/";
}
