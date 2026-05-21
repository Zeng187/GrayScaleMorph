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
        double wP_kap;
        double wP_lam;
        double penalty_threshold;
        double betaP;
        int    stage_iter;
        int    warmup_stages;
        double warmup_reg_decay;
        double mgda_reg_decay;
        double mgda_alpha_start;   // per-stage alpha schedule, from start@stage 0
        double mgda_alpha_end;     // to end@stage N-1.  Negative -> fallback to closed-form alpha.
        double mgda_alpha_decay_exp;  // schedule shape: alpha_k = end + (start - end) * (1 - k/(N-1))^exp
                                      // exp = 1.0 -> linear; exp > 1 -> stay near start longer (favor d_F).
        std::string morph_method;
        // Forward-only knobs (default false → legacy patch_0_ + target-V behavior).
        // WholeMeshMode  : read {model}_param.obj + {model}_bound_center.txt
        //                  instead of patch_0_P.obj + patch_0_bound_center.txt.
        // InitFromParam  : skip reading target V; initialize Newton with
        //                  P + tiny random z perturbation. Use this for
        //                  forward-only synthetic shapes that have no
        //                  reference 3D target.
        bool   whole_mesh_mode;
        bool   init_from_param;
    } RuntimeSetting;

    // --------- Convenience: full material .json path ---------
    // Composed as PathSetting.MaterialsDir + ModelSetting.MaterialName + ".json".
    std::string materialJsonPath() const;
};
