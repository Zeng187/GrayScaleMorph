#pragma once


#include <vector>
#include <string>


class Config
{
public:
    Config(const std::string& filePath);

    struct
    {
        std::string ResourcePath;
        std::string MaterialPath;
    }ResourceSetting;
    struct
    {
        std::string ModelName;
        std::string Postfix;
        std::string InputPath;
    }ModelSetting;

    struct
    {
        std::string OutputPath;
        std::string Mode;
    }OutputSetting;
};