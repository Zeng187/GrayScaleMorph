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

    struct {
        int    Platewidth;
        int    MaxIter;
        int    nFmin;
        double epsilon;
        double wM;
        double wL;
        double wM_kap;
        double wL_kap;
        double wM_lam;
        double wL_lam;
        double w_s;
        double w_b;
        double wP_kap;
        double wP_lam;
        double penalty_threshold;
        double betaP;
    } RuntimeSetting;
};