#include "config.hpp"
#include <fstream>
#include <filesystem>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

using json = nlohmann::json;

namespace {

json load_json_file(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        spdlog::error("Cannot open config file: {}", path);
        std::exit(1);
    }
    json j;
    f >> j;
    return j;
}

} // namespace

Config::Config(const std::string& filePath) {
    json j = load_json_file(filePath);

    // -------- PathSetting (from PathConfig -> path.json) --------
    if (!j.contains("PathConfig")) {
        spdlog::error("cfg.json missing required 'PathConfig' field "
                      "(should point at Resources/0_setup/path.json).");
        std::exit(1);
    }
    const std::string pathJsonRef = j["PathConfig"][0];
    json pj = load_json_file(pathJsonRef);
    if (!pj.contains("Paths")) {
        spdlog::error("PathConfig file '{}' missing 'Paths' object.", pathJsonRef);
        std::exit(1);
    }
    const auto& p = pj["Paths"];
    PathSetting.ConesDir      = p["ConesDir"][0];
    PathSetting.MeshesDir     = p["MeshesDir"][0];
    PathSetting.MeshesPostDir = p["MeshesPostDir"][0];
    PathSetting.PoseDir       = p["PoseDir"][0];
    PathSetting.RidgeDir      = p["RidgeDir"][0];
    PathSetting.ConfigDir     = p["ConfigDir"][0];
    PathSetting.InitialDir    = p["InitialDir"][0];
    PathSetting.MaterialsDir  = p["MaterialsDir"][0];
    PathSetting.SegmentDir    = p["SegmentDir"][0];
    // MassDir: optional (back-compat) — per-vertex mass weights from 1_post_cut.
    PathSetting.MassDir       = p.value("MassDir", nlohmann::json::array({"../../Resources/1_mass/"}))[0];
    PathSetting.DesignDir     = p["DesignDir"][0];
    PathSetting.TargetDir     = p["TargetDir"][0];
    PathSetting.MorphDir      = p["MorphDir"][0];
    PathSetting.MorphLogsDir  = p.value("MorphLogsDir", nlohmann::json::array({"../../Resources/2_morphlogs/"}))[0];
    PathSetting.ParamDir      = p["ParamDir"][0];
    PathSetting.MorphInitDir  = p["MorphInitDir"][0];
    PathSetting.CondDir       = p["CondDir"][0];
    PathSetting.ForwardDir    = p["ForwardDir"][0];
    PathSetting.FigsDir       = p["FigsDir"][0];
    spdlog::info("PathConfig loaded from {}", pathJsonRef);

    // -------- DeviceSetting (sibling of path.json, i.e. <PathConfig dir>/device.json) --------
    const std::string deviceJsonPath =
        std::filesystem::path(pathJsonRef).parent_path().string() + "/device.json";
    json dj = load_json_file(deviceJsonPath);
    DeviceSetting.platewidth = dj["platewidth"];
    spdlog::info("DeviceSetting loaded from {} (platewidth={})",
                 deviceJsonPath, DeviceSetting.platewidth);

    // -------- Model --------
    auto& m = j["Model"];
    ModelSetting.ModelName    = m["ModelName"][0];
    ModelSetting.Postfix      = m.value("Postfix",      json::array({".obj"}))[0];
    ModelSetting.MaterialName = m.value("MaterialName", json::array({"largedeform-material"}))[0];
    ModelSetting.DesignName   = m.value("DesignName",   json::array({""}))[0];

    // -------- Runtime hyperparameters --------
    auto& rt = j["RuntimeSettings"];
    RuntimeSetting.MaxIter            = rt["MaxIter"][0];
    RuntimeSetting.nFmin              = rt["nFmin"][0];
    RuntimeSetting.epsilon            = rt["epsilon"][0];
    RuntimeSetting.wM_kap             = rt["wM_kap"][0];
    RuntimeSetting.wL_kap             = rt["wL_kap"][0];
    RuntimeSetting.wM_lam             = rt["wM_lam"][0];
    RuntimeSetting.wL_lam             = rt["wL_lam"][0];
    RuntimeSetting.w_s                = rt["w_s"][0];
    RuntimeSetting.w_b                = rt["w_b"][0];
    // wP_lam / wP_kap: prefer the new keys; fall back to unified "wP" (used
    // for both) or legacy "wP_kap" only key for old cfgs.
    {
        const double default_wP = rt.contains("wP")     ? double(rt["wP"][0])
                                : rt.contains("wP_kap") ? double(rt["wP_kap"][0])
                                : 0.0;
        RuntimeSetting.wP_lam = rt.value("wP_lam", nlohmann::json::array({default_wP}))[0];
        RuntimeSetting.wP_kap = rt.value("wP_kap", nlohmann::json::array({default_wP}))[0];
    }
    RuntimeSetting.wM_P  = rt.value("wM_P",  nlohmann::json::array({1e-4}))[0];
    RuntimeSetting.wL_P  = rt.value("wL_P",  nlohmann::json::array({1e-3}))[0];
    RuntimeSetting.wSLIM = rt.value("wSLIM", nlohmann::json::array({0.0}))[0];
    // wSLIM_decay defaults to 1.0 (no decay -> back-compat with old cfgs).
    RuntimeSetting.wSLIM_decay = rt.value("wSLIM_decay", nlohmann::json::array({1.0}))[0];
    // wSLIM_final defaults to the initial wSLIM (back-compat).
    RuntimeSetting.wSLIM_final = rt.value("wSLIM_final",
                                          nlohmann::json::array({RuntimeSetting.wSLIM}))[0];
    // run_stage_optp defaults to true (current behaviour).  Set false to skip
    // the per-stage SGN OptP inside the BCD loop -- P stays at P_anchor for
    // the entire BCD pass; only (kappa, lambda) are updated.  Useful to test
    // whether the OptP step is responsible for seam-quality issues.
    RuntimeSetting.run_stage_optp = rt.value("run_stage_optp",
                                             nlohmann::json::array({true}))[0];
    // run_final_optp defaults to true (current behaviour).  Set false to skip
    // the post-BCD SGN OptP pass on snapped material (P freezes at the best
    // snapshot from the BCD loop).
    RuntimeSetting.run_final_optp = rt.value("run_final_optp",
                                             nlohmann::json::array({true}))[0];
    // optp_uniform_mass defaults to false (back-compat).  When true, every OptP
    // SGN call (both per-stage and FinalSnap) uses a uniform per-vertex mass
    // (1/nV broadcast across xyz), so P-layout updates are not driven by the
    // segmentation-aware mass discontinuity at seams.  OptKap / OptLam still
    // use the segmentation-aware `masses`.
    RuntimeSetting.optp_uniform_mass = rt.value("optp_uniform_mass",
                                                nlohmann::json::array({false}))[0];
    // min_angle_deg defaults to 0 (disabled) for back-compat.  A typical value
    // is 5-10 degrees; trial P that creates a thinner face is rejected by the
    // line search via +inf return from the SGN OptP `distance` lambda.
    RuntimeSetting.min_angle_deg = rt.value("min_angle_deg",
                                            nlohmann::json::array({0.0}))[0];
    // patch_id defaults to 0 (back-compat with single-patch single-mesh runs).
    RuntimeSetting.patch_id    = rt.value("patch_id", nlohmann::json::array({0}))[0];
    RuntimeSetting.penalty_threshold  = rt["penalty_threshold"][0];
    RuntimeSetting.betaP              = rt["betaP"][0];
    RuntimeSetting.snap_before_P      = rt.value("snap_before_P",    nlohmann::json::array({false}))[0];
    RuntimeSetting.stage_iter         = rt.value("stage_iter",       nlohmann::json::array({5}))[0];
    RuntimeSetting.stage_continuous   = rt.value("stage_continuous", nlohmann::json::array({RuntimeSetting.stage_iter / 2}))[0];
    // wP_lam_growth_factor / wP_kap_growth_factor: prefer the per-direction
    // keys; fall back to unified "wP_growth_factor" or default 1.0.
    {
        const double default_g = rt.value("wP_growth_factor", nlohmann::json::array({1.0}))[0];
        RuntimeSetting.wP_lam_growth_factor =
            rt.value("wP_lam_growth_factor", nlohmann::json::array({default_g}))[0];
        RuntimeSetting.wP_kap_growth_factor =
            rt.value("wP_kap_growth_factor", nlohmann::json::array({default_g}))[0];
    }
    RuntimeSetting.morph_method       = rt.value("morph_method",     nlohmann::json::array({std::string("homotopy")}))[0].get<std::string>();
    RuntimeSetting.joint_penalty_alpha = rt.value("joint_penalty_alpha", nlohmann::json::array({1.0}))[0];
}

std::string Config::materialJsonPath() const
{
    return PathSetting.MaterialsDir + ModelSetting.MaterialName + ".json";
}
