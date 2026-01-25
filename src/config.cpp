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

    this->ResourceSetting.ResourcePath = j["Resource"]["ResourcePath"][0].get<std::string>();
	this->ResourceSetting.MaterialPath = j["Resource"]["MaterialPath"][0].get<std::string>();

    this->ModelSetting.ModelName = j["Model"]["ModelName"][0].get<std::string>();
    this->ModelSetting.Postfix = j["Model"]["Postfix"][0].get<std::string>();
    this->ModelSetting.InputPath = j["Model"]["InputPath"][0].get<std::string>();

    this->OutputSetting.OutputPath = j["OutputSettings"]["OutputPath"][0].get<std::string>();
    this->OutputSetting.Mode = j["OutputSettings"]["Mode"][0].get<std::string>();

    spdlog::info("Configure file found in {0}", filePath);
}