#include "material.hpp"
#include <fstream>
#include <filesystem>
#include<algorithm>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include<Eigen/Core>
#include <Eigen/Dense>

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

    // Scheme-A calibrated curvature coefficient (optional).
    // 0.0 sentinel => not provided => fall back to modulus-weighted physics.
    kappa_factor = m.contains("kappa_factor") ? m["kappa_factor"].get<double>() : 0.0;

    assert(youngs_modulus.size() == strech_ratio.size());
    assert(strech_ratio.size() ==count);
    // process
    for(auto &s: strech_ratio)
        s = s * 0.01;

	t_vals.resize(count);
    for (int i = 0; i < count; i++)
        t_vals[i] = (double)i / (count - 1);

    spdlog::info("Material file found in {0}",filePath);
}


void Grayscale_Material::ComputeMaterialCurve()
{
    if (count <= 1) return;

    // Natural cubic spline: second derivative = 0 at both endpoints.
    // Interpolates the 7 measured (t_i, value_i) points exactly.
    m_strain_curve.curve.set_boundary(
        tk::spline::second_deriv, 0.0,
        tk::spline::second_deriv, 0.0);
    m_strain_curve.curve.set_points(t_vals, strech_ratio, tk::spline::cspline);

    m_moduls_curve.curve.set_boundary(
        tk::spline::second_deriv, 0.0,
        tk::spline::second_deriv, 0.0);
    m_moduls_curve.curve.set_points(t_vals, youngs_modulus, tk::spline::cspline);

    spdlog::info("Material curves built via natural cubic spline over {} measured points", count);
}




ActiveComposite::ActiveComposite(const std::string& filePath):Grayscale_Material(filePath)
{
    double strain_min = *std::min_element(strech_ratio.begin(), strech_ratio.end());
    double strain_max = *std::max_element(strech_ratio.begin(), strech_ratio.end());

    range_lam = double2{ 1 + strain_min,1 + strain_max };

    // Curvature range uses the calibrated scheme-A factor (kappa = factor*(s1-s2)/h);
    // fall back to the legacy 1.5 coefficient when no kappa_factor is provided.
    const double kfac = (kappa_factor != 0.0) ? kappa_factor : 1.5;
    double _kappa_ = kfac * (strain_max - strain_min) / thickness;

    range_kap = double2{ -_kappa_, _kappa_};

    LoadEsurface(filePath);
}


void ActiveComposite::ComputeFeasibleVals()
{
    fesasible_cnt = count * count;
    feasible_t_vals.resize(fesasible_cnt);
    feasible_lamb.resize(fesasible_cnt);
    feasible_kapp.resize(fesasible_cnt);
    feasible_modl.resize(fesasible_cnt);

    for(int j = 0; j<count;j++)
    {
        for(int i = 0;i<count; i++)
        {
            int id = j * count + i;
            double t1 = (double) i /(double)(count -1);
            double t2 = (double) j /(double)(count -1);

            double lam = compute_lamb_d(m_strain_curve, m_moduls_curve, t1, t2);
            double kap = compute_curv_d(m_strain_curve, m_moduls_curve, thickness, t1, t2, kappa_factor);
            double mol = compute_modu_d(m_moduls_curve,t1,t2);

            feasible_t_vals[id]=std::pair<double,double>(t1,t2);
            feasible_lamb[id] = lam;
            feasible_kapp[id] = kap;
            feasible_modl[id] = mol;
        }
    }

}


void ActiveComposite::LoadEsurface(const std::string& filePath)
{
    // filePath is the raw material JSON (e.g. ".../grayscale-material.json").
    // The TPS surface lives next to it as
    //   ".../grayscale-material-modulus_surface_tps.json".
    std::filesystem::path raw_path(filePath);
    auto stem = raw_path.stem().string();
    auto surface_path =
        (raw_path.parent_path() / (stem + "-modulus_surface_tps.json")).string();

    auto install_identity_fallback = [&]() {
        // Identity TPS: no radial basis terms, affine = [1, 0, 0] -> E == 1.
        m_E_surface.anchors_lambda_hat.clear();
        m_E_surface.anchors_kappa_sq_hat.clear();
        m_E_surface.tps_weights = std::vector<double>{1.0, 0.0, 0.0};
        m_E_surface.lambda_mid = 0.0;
        m_E_surface.lambda_half_range = 1.0;
        m_E_surface.kappa_sq_max = 1.0;
        m_E_surface.loaded = false;
    };

    std::ifstream sfile(surface_path);
    if (!sfile.is_open()) {
        spdlog::warn("E(lambda, kappa) TPS surface NOT found: {} -- falling back to identity E=1",
                     surface_path);
        install_identity_fallback();
        return;
    }

    json sj;
    sfile >> sj;

    try {
        auto& norm = sj["normalization"];
        m_E_surface.lambda_mid        = norm["lambda_mid"].get<double>();
        m_E_surface.lambda_half_range = norm["lambda_half_range"].get<double>();
        m_E_surface.kappa_sq_max      = norm["kappa_sq_max"].get<double>();

        m_E_surface.anchors_lambda_hat   = sj["anchors_lambda_hat"].get<std::vector<double>>();
        m_E_surface.anchors_kappa_sq_hat = sj["anchors_kappa_sq_hat"].get<std::vector<double>>();
        m_E_surface.tps_weights          = sj["tps_weights"].get<std::vector<double>>();

        const int N = static_cast<int>(m_E_surface.anchors_lambda_hat.size());
        const int expected_N = sj.contains("anchor_count")
            ? sj["anchor_count"].get<int>()
            : N;

        if (static_cast<int>(m_E_surface.anchors_kappa_sq_hat.size()) != N ||
            N != expected_N) {
            spdlog::error("modulus_surface_tps.json: anchor arrays inconsistent "
                          "(lambda_hat={}, kappa_sq_hat={}, anchor_count={})",
                          N,
                          (int)m_E_surface.anchors_kappa_sq_hat.size(),
                          expected_N);
            exit(1);
        }
        if (static_cast<int>(m_E_surface.tps_weights.size()) != N + 3) {
            spdlog::error("modulus_surface_tps.json: tps_weights size {} != N+3 = {}",
                          (int)m_E_surface.tps_weights.size(), N + 3);
            exit(1);
        }

        m_E_surface.loaded = true;

        // Sanity log: evaluate at (lambda = lambda_mid, kappa = 0) so lh=0, ksh=0.
        // The radial basis at the centre is non-zero (sum of phi(||p_k||) terms),
        // so this exercises the real evaluator path rather than just the affine bias.
        const double E_center = compute_E_lk<double>(
            m_E_surface, m_E_surface.lambda_mid, 0.0);
        spdlog::info("Loaded E(lambda, kappa) TPS surface from {} (N={} anchors, "
                     "E at lambda_mid,kappa=0 = {} MPa)",
                     surface_path, N, E_center);
    } catch (const std::exception& ex) {
        spdlog::error("modulus_surface_tps.json: parse error: {} -- falling back to identity E=1",
                      ex.what());
        install_identity_fallback();
    }
}
