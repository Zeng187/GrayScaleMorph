#include "config.hpp"

#include <cstdlib>
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
            auto inputPath = jsonGet<std::string>(sec, "InputPath");
            auto postfix   = jsonGet<std::string>(sec, "Postfix");
            model.mesh_path = inputPath + model.name + postfix;
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
        if (sec.contains("path"))
            segment.path = jsonGet<std::string>(sec, "path");
        else
            segment.path = jsonGet<std::string>(sec, "SegmentPath");        // legacy

        segment.method = jsonGetOr<std::string>(sec, "method", "");
        if (segment.method.empty())
            segment.method = jsonGetOr<std::string>(sec, "DistortionMethod", "");

        segment.plan = jsonGetOr<std::string>(sec, "plan", "");
        if (segment.plan.empty())
            segment.plan = jsonGetOr<std::string>(sec, "Plan", "");
    }

    // ── output ───────────────────────────────────────────────────────
    {
        const auto& sec = findSection(j, {"output", "OutputSettings"});
        if (sec.contains("output_path"))
            output.output_path = jsonGet<std::string>(sec, "output_path");
        else
            output.output_path = jsonGet<std::string>(sec, "OutputPath");   // legacy

        output.param_path = jsonGetOr<std::string>(sec, "param_path", "../Resources/param/");

        if (sec.contains("morph_path"))
            output.morph_path = jsonGet<std::string>(sec, "morph_path");
        else
            output.morph_path = jsonGet<std::string>(sec, "MorphPath");     // legacy

        if (sec.contains("design_path"))
            output.design_path = jsonGet<std::string>(sec, "design_path");
        else
            output.design_path = jsonGet<std::string>(sec, "DesignPath");   // legacy
    }

    // ── solver ───────────────────────────────────────────────────────
    {
        const auto& sec = findSection(j, {"solver", "RuntimeSettings"});
        solver.platewidth        = jsonGetOr(sec, "platewidth",        jsonGetOr(sec, "Platewidth", solver.platewidth));
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
        solver.betaP             = jsonGetOr(sec, "betaP",            solver.betaP);
    }

    spdlog::info("Config: model='{}', mesh='{}'", model.name, model.mesh_path);
}

// ═══════════════════════════════════════════════════════════════════════
// Path builders
// ═══════════════════════════════════════════════════════════════════════

std::string Config::segmentDir(const std::string& modelName) const
{
    std::string dirName = modelName;
    if (!segment.method.empty())
        dirName += "_" + segment.method;
    if (!segment.plan.empty())
        dirName += "_" + segment.plan;
    return segment.path + dirName + "/";
}

std::string Config::paramDir() const
{
    return output.param_path + model.name + "/";
}

std::string Config::morphDir() const
{
    return output.morph_path + model.name + "/";
}

std::string Config::designDir() const
{
    return output.design_path + model.name + "/";
}
