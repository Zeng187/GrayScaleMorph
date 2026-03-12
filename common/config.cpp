#include "config.hpp"
#include <fstream>
#include <filesystem>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

using json = nlohmann::json;

Config::Config(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        spdlog::error("Configure file not found!");
        exit(1);
    }

    json j;
    file >> j;

    auto& r = j["Resource"];
    ResourceSetting.ResourcePath = r["ResourcePath"][0];
    ResourceSetting.MaterialPath = r["MaterialPath"][0];

    auto& m = j["Model"];
    ModelSetting.ModelName = m["ModelName"][0];
    ModelSetting.Postfix = m["Postfix"][0];
    ModelSetting.InputPath = m["InputPath"][0];

    auto& o = j["OutputSettings"];
    OutputSetting.OutputPath = o["OutputPath"][0];
    OutputSetting.MorphPath = o["MorphPath"][0];
    OutputSetting.DesignPath = o["DesignPath"][0];
    OutputSetting.MetricsPath = o["MetricsPath"][0];
    OutputSetting.Mode = o["Mode"][0];

    auto& rt = j["RuntimeSettings"];
    RuntimeSetting.Platewidth = rt["Platewidth"][0];
    RuntimeSetting.MaxIter = rt["MaxIter"][0];
    RuntimeSetting.nFmin = rt["nFmin"][0];
    RuntimeSetting.epsilon = rt["epsilon"][0];
    RuntimeSetting.wM = rt["wM"][0];
    RuntimeSetting.wL = rt["wL"][0];
    RuntimeSetting.wM_kap = rt["wM_kap"][0];
    RuntimeSetting.wL_kap = rt["wL_kap"][0];
    RuntimeSetting.wM_lam = rt["wM_lam"][0];
    RuntimeSetting.wL_lam = rt["wL_lam"][0];
    RuntimeSetting.w_s = rt["w_s"][0];
    RuntimeSetting.w_b = rt["w_b"][0];
    RuntimeSetting.wP_kap = rt["wP_kap"][0];
    RuntimeSetting.wP_lam = rt["wP_lam"][0];
    RuntimeSetting.penalty_threshold = rt["penalty_threshold"][0];
    RuntimeSetting.betaP = rt["betaP"][0];
}