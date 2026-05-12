#include "setup.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr double kThicknessTolerance = 1.0e-9;

template <typename T>
T jsonRequired(const json& obj, const char* key, const std::string& source)
{
    if (!obj.contains(key)) {
        throw std::runtime_error("Missing required key '" + std::string(key)
                                 + "' in " + source);
    }
    return obj.at(key).get<T>();
}

fs::path resolveRelativeToCfg(const fs::path& cfg_path, const fs::path& raw)
{
    if (raw.is_absolute()) return raw.lexically_normal();
    const auto cfg_abs = fs::absolute(cfg_path);
    return fs::absolute(cfg_abs.parent_path() / raw).lexically_normal();
}

json readJsonFile(const fs::path& path, const char* what)
{
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error(std::string("Cannot open ") + what + ": "
                                 + path.string());
    }
    json j;
    in >> j;
    if (!j.is_object()) {
        throw std::runtime_error(std::string(what) + " must be a JSON object: "
                                 + path.string());
    }
    return j;
}

std::optional<fs::path> explicitSetupPath(const json& cfg_root)
{
    if (cfg_root.contains("setup_path")) {
        return fs::path(cfg_root.at("setup_path").get<std::string>());
    }
    if (cfg_root.contains("setup") && cfg_root.at("setup").is_object()) {
        const auto& s = cfg_root.at("setup");
        if (s.contains("path")) {
            return fs::path(s.at("path").get<std::string>());
        }
    }
    return std::nullopt;
}

bool isSchemaVersionSupported(const std::string& version)
{
    // Major-version compatibility: accept any "1.x.y".
    return version.rfind("1.", 0) == 0;
}

void requirePositive(double value, const char* name, const fs::path& source)
{
    if (!(value > 0.0)) {
        throw std::runtime_error(std::string("Global setup field '") + name
                                 + "' must be > 0 (got "
                                 + std::to_string(value) + ") in "
                                 + source.string());
    }
}

double extractMaterialThickness(const json& material_root, const fs::path& path)
{
    if (material_root.contains("thickness")) {
        return material_root.at("thickness").get<double>();
    }
    if (material_root.contains("materials")
        && material_root.at("materials").is_object()
        && material_root.at("materials").contains("thickness"))
    {
        return material_root.at("materials").at("thickness").get<double>();
    }
    throw std::runtime_error("Material JSON has no 'thickness' or "
                             "'materials.thickness': " + path.string());
}

} // namespace

fs::path resolveGlobalSetupPath(const json& cfg_root, const fs::path& cfg_path)
{
    if (const auto raw = explicitSetupPath(cfg_root)) {
        const auto resolved = resolveRelativeToCfg(cfg_path, *raw);
        if (!fs::is_regular_file(resolved)) {
            throw std::runtime_error("Explicit setup_path does not exist: "
                                     + resolved.string());
        }
        return resolved;
    }

    auto dir = fs::absolute(cfg_path).parent_path();
    while (true) {
        const auto candidate = dir / "Resources" / "setup" / "global.json";
        if (fs::is_regular_file(candidate)) {
            return candidate.lexically_normal();
        }
        const auto parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }

    throw std::runtime_error(
        "Could not locate Resources/setup/global.json walking upward from: "
        + fs::absolute(cfg_path).parent_path().string());
}

void validateGlobalSetup(const GlobalSetup& setup)
{
    if (!isSchemaVersionSupported(setup.schema_version)) {
        throw std::runtime_error("Unsupported global setup schema_version '"
                                 + setup.schema_version + "' in "
                                 + setup.source_path.string()
                                 + " (expected 1.x.y)");
    }
    requirePositive(setup.platewidth,    "platewidth",    setup.source_path);
    requirePositive(setup.thickness,     "thickness",     setup.source_path);
    requirePositive(setup.slice_height,  "slice_height",  setup.source_path);

    if (!(setup.poisson_ratio > -1.0 && setup.poisson_ratio <= 0.5)) {
        throw std::runtime_error("Global setup field 'poisson_ratio' must "
                                 "satisfy -1 < nu <= 0.5 (got "
                                 + std::to_string(setup.poisson_ratio)
                                 + ") in " + setup.source_path.string());
    }
}

void validateThicknessAgainstMaterial(const GlobalSetup& setup,
                                      const fs::path& material_path,
                                      const fs::path& cfg_path)
{
    const auto resolved = resolveRelativeToCfg(cfg_path, material_path);
    const auto root = readJsonFile(resolved, "material JSON");
    const double material_thickness = extractMaterialThickness(root, resolved);

    if (std::abs(material_thickness - setup.thickness) > kThicknessTolerance) {
        throw std::runtime_error(
            "Thickness mismatch: setup.thickness=" + std::to_string(setup.thickness)
            + " vs material.thickness=" + std::to_string(material_thickness)
            + " (in " + resolved.string() + "). "
            "Either re-run MaterialGen with the new thickness or restore the old value.");
    }
}

GlobalSetup loadGlobalSetup(const json& cfg_root,
                            const LoadGlobalSetupOptions& options)
{
    const auto setup_path = resolveGlobalSetupPath(cfg_root, options.cfg_path);
    const auto root = readJsonFile(setup_path, "global setup JSON");
    const std::string source = setup_path.string();

    GlobalSetup setup;
    setup.schema_version = jsonRequired<std::string>(root, "schema_version", source);
    setup.platewidth     = jsonRequired<double>     (root, "platewidth",     source);
    setup.thickness      = jsonRequired<double>     (root, "thickness",      source);
    setup.slice_height   = jsonRequired<double>     (root, "slice_height",   source);
    setup.poisson_ratio  = jsonRequired<double>     (root, "poisson_ratio",  source);
    setup.source_path    = setup_path;

    validateGlobalSetup(setup);

    if (options.material_path.has_value()) {
        validateThicknessAgainstMaterial(setup, *options.material_path,
                                         options.cfg_path);
    }

    spdlog::info("Loaded global setup from {}: "
                 "platewidth={}, thickness={}, slice_height={}, poisson_ratio={}",
                 setup.source_path.string(),
                 setup.platewidth, setup.thickness,
                 setup.slice_height, setup.poisson_ratio);
    return setup;
}
