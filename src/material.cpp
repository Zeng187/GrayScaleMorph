#include "material.hpp"
#include <fstream>
#include <filesystem>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

using json = nlohmann::json;

Grayscale_Material::Grayscale_Material(const std::string& filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        spdlog::error("Material file not found!");
        exit(1);
    }

    json j;
    file >> j;


    auto& m = j["materials"];

    name = m["name"];
    description = m["description"];
    count = m["count"];
    youngs_modulus = m["youngs_modulus"].get<std::vector<double>>();
    strech_ratio = m["strech_ratio"].get<std::vector<double>>();

    spdlog::info("Material file found in {0}",filePath);
}

