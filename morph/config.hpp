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
    // Mirrors Resources/0_setup/path.json verbatim (16 keys).
    struct
    {
        std::string ConesDir;
        std::string MeshesDir;
        std::string MeshesPostDir;
        std::string PoseDir;
        std::string RidgeDir;
        std::string ConfigDir;
        std::string InitialDir;
        std::string MaterialsDir;
        std::string SegmentDir;
        std::string DesignDir;
        std::string TargetDir;
        std::string MorphDir;
        std::string MorphLogsDir;
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
        std::string DesignName;     // optional: base filename (no .txt) inside DesignDir/{model}/.
                                    //   Empty -> default per-patch convention "patch_{i}_material".
                                    //   Set   -> Forward reads "{DesignName}.txt" and writes
                                    //            "{DesignName}_pred.obj" instead.
    } ModelSetting;

    // --------- Physical device / process settings (from SetupDir + "device.json") ---------
    struct {
        double platewidth;
    } DeviceSetting;

    // --------- Algorithm hyperparameters ----------
    struct {
        int    MaxIter;
        int    nFmin;
        double epsilon;
        double wM_kap;
        double wL_kap;
        double wM_lam;
        double wL_lam;
        double w_s;
        double w_b;
        double wP_lam;                  // A in penalty = A*(lam^2-lam_i^2)^2 + B*(kap-kap_i)^2
        double wP_kap;                  // B
        double wM_P;                    // P-anchor mass weight (||P - P_anchor||^2)
        double wL_P;                    // P-smoothness Laplacian weight
        double wSLIM;                   // SLIM symmetric-Dirichlet barrier weight (foldover prevention)
        double penalty_threshold;
        double betaP;
        bool   snap_before_P;           // hard-snap (lambda, kappa) to nearest feasible before each P-update
        int    stage_iter;              // outer alternating-stage count
        int    stage_continuous;        // first N stages: OptP uses continuous material; remaining stages: OptP uses snap material
        double wP_lam_growth_factor;    // initial (1+factor) homotopy step for A=wP_lam
        double wP_kap_growth_factor;    // initial (1+factor) homotopy step for B=wP_kap
        double joint_penalty_alpha;     // legacy field, unused (kept for cfg back-compat)
        std::string morph_method;       // subdir name under MorphLogsDir; e.g. "homotopy", "mgda"
    } RuntimeSetting;

    // --------- Convenience: full material .json path ---------
    // Composed as PathSetting.MaterialsDir + ModelSetting.MaterialName + ".json".
    std::string materialJsonPath() const;
};
