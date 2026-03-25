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
        std::string SegmentPath;
        std::string DistortionMethod;
        std::string Plan;
    }ResourceSetting;

    /// Build the segment directory for a given model name.
    /// Appends _{method} if DistortionMethod is non-empty, and _{plan} if Plan is non-empty.
    /// e.g. SegmentPath + "bird_yamabe_planB/" vs SegmentPath + "bird/"
    std::string segmentDir(const std::string& modelName) const;
    struct
    {
        std::string ModelName;
        std::string Postfix;
        std::string InputPath;
    }ModelSetting;

    struct
    {
        std::string OutputPath;
        std::string MorphPath;
        std::string DesignPath;
        std::string MetricsPath;
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