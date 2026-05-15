#pragma once


#include <vector>
#include <string>


class Config
{
public:
    Config(const std::string& filePath);

    // --------- PathSetting: populated by reading PathConfig file ----------
    // All paths are interpreted as relative to the binary's working
    // directory (typically <S2_GrayScaleMorph>/<module>/, so a value of
    // "../../Resources/0_meshes/" resolves to the project Resources tree).
    // Mirrors Resources/0_setup/path.json verbatim (18 keys).
    struct
    {
        std::string ConesDir;
        std::string MeshesDir;
        std::string MeshesPostDir;
        std::string PoseDir;
        std::string PostDir;
        std::string RidgeDir;
        std::string SetupDir;
        std::string YoshizawaDir;
        std::string ConfigDir;
        std::string InitialDir;
        std::string MaterialsDir;
        std::string SegmentDir;
        std::string DesignDir;
        std::string MorphDir;
        std::string ParamDir;
        std::string CondDir;
        std::string ForwardDir;
        std::string FigsDir;
    } PathSetting;

    // --------- Model selection ----------
    struct
    {
        std::string ModelName;
        std::string Postfix;
        std::string MaterialName;   // base filename (no .json), used with PathSetting.MaterialsDir
    } ModelSetting;

    // --------- Optional: per-binary output override (rarely used) ---------
    struct
    {
        std::string Mode;
    } OutputSetting;

    // --------- Algorithm hyperparameters ----------
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

    // --------- Convenience: full material .json path ---------
    // Composed as PathSetting.MaterialsDir + ModelSetting.MaterialName + ".json".
    std::string materialJsonPath() const;
};
