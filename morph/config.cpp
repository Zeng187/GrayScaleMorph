#include "config.hpp"
#include <fstream>
#include <filesystem>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

using json = nlohmann::json;

namespace {

json load_json_file(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        spdlog::error("Cannot open config file: {}", path);
        std::exit(1);
    }
    json j;
    f >> j;
    return j;
}

} // namespace

Config::Config(const std::string& filePath) {
    json j = load_json_file(filePath);

    // -------- PathSetting (from PathConfig -> path.json) --------
    if (!j.contains("PathConfig")) {
        spdlog::error("cfg.json missing required 'PathConfig' field "
                      "(should point at Resources/0_setup/path.json).");
        std::exit(1);
    }
    const std::string pathJsonRef = j["PathConfig"][0];
    json pj = load_json_file(pathJsonRef);
    if (!pj.contains("Paths")) {
        spdlog::error("PathConfig file '{}' missing 'Paths' object.", pathJsonRef);
        std::exit(1);
    }
    const auto& p = pj["Paths"];
    PathSetting.ConesDir      = p["ConesDir"][0];
    PathSetting.MeshesDir     = p["MeshesDir"][0];
    PathSetting.MeshesPostDir = p["MeshesPostDir"][0];
    PathSetting.PoseDir       = p["PoseDir"][0];
    PathSetting.RidgeDir      = p["RidgeDir"][0];
    PathSetting.ConfigDir     = p["ConfigDir"][0];
    PathSetting.InitialDir    = p["InitialDir"][0];
    PathSetting.MaterialsDir  = p["MaterialsDir"][0];
    PathSetting.SegmentDir    = p["SegmentDir"][0];
    PathSetting.DesignDir     = p["DesignDir"][0];
    PathSetting.TargetDir     = p["TargetDir"][0];
    PathSetting.MorphDir      = p["MorphDir"][0];
    PathSetting.ParamDir      = p["ParamDir"][0];
    PathSetting.CondDir       = p["CondDir"][0];
    PathSetting.ForwardDir    = p["ForwardDir"][0];
    PathSetting.FigsDir       = p["FigsDir"][0];
    spdlog::info("PathConfig loaded from {}", pathJsonRef);

    // -------- DeviceSetting (sibling of path.json, i.e. <PathConfig dir>/device.json) --------
    const std::string deviceJsonPath =
        std::filesystem::path(pathJsonRef).parent_path().string() + "/device.json";
    json dj = load_json_file(deviceJsonPath);
    DeviceSetting.platewidth = dj["platewidth"];
    spdlog::info("DeviceSetting loaded from {} (platewidth={})",
                 deviceJsonPath, DeviceSetting.platewidth);

    // -------- Model --------
    auto& m = j["Model"];
    ModelSetting.ModelName    = m["ModelName"][0];
    ModelSetting.Postfix      = m.value("Postfix",      json::array({".obj"}))[0];
    ModelSetting.MaterialName = m.value("MaterialName", json::array({"grayscale-material"}))[0];
    ModelSetting.DesignName   = m.value("DesignName",   json::array({""}))[0];

    // -------- Runtime hyperparameters --------
    auto& rt = j["RuntimeSettings"];
    RuntimeSetting.MaxIter           = rt["MaxIter"][0];
    RuntimeSetting.nFmin             = rt["nFmin"][0];
    RuntimeSetting.epsilon           = rt["epsilon"][0];
    RuntimeSetting.wM                = rt["wM"][0];
    RuntimeSetting.wL                = rt["wL"][0];
    RuntimeSetting.wM_kap            = rt["wM_kap"][0];
    RuntimeSetting.wL_kap            = rt["wL_kap"][0];
    RuntimeSetting.wM_lam            = rt["wM_lam"][0];
    RuntimeSetting.wL_lam            = rt["wL_lam"][0];
    RuntimeSetting.w_s               = rt["w_s"][0];
    RuntimeSetting.w_b               = rt["w_b"][0];
    RuntimeSetting.wP_kap            = rt["wP_kap"][0];
    RuntimeSetting.wP_lam            = rt["wP_lam"][0];
    RuntimeSetting.penalty_threshold = rt["penalty_threshold"][0];
    RuntimeSetting.betaP             = rt["betaP"][0];
}

std::string Config::materialJsonPath() const
{
    return PathSetting.MaterialsDir + ModelSetting.MaterialName + ".json";
}
