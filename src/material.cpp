#include "material.hpp"
#include <fstream>
#include <filesystem>
#include<algorithm>

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
    thickness = m["thickness"];
    count = m["count"];
    youngs_modulus = m["youngs_modulus"].get<std::vector<double>>();
    strech_ratio = m["strech_ratio"].get<std::vector<double>>();

    // process
    for(auto &s: strech_ratio)
        s = s * 0.01;

    spdlog::info("Material file found in {0}",filePath);
}

ActiveComposite::ActiveComposite(const std::string& filePath):Grayscale_Material(filePath)
{
    double strain_min = *std::min_element(strech_ratio.begin(), strech_ratio.end());
    double strain_max = *std::max_element(strech_ratio.begin(), strech_ratio.end());

    range_lam = double2{ 1 + strain_min,1 + strain_max };

    double _kappa_ = 1.5 * (strain_max - strain_min) / thickness;

    range_kap = double2{ -_kappa_, _kappa_};
}

