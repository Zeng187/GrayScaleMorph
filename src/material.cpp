#include "material.hpp"
#include <fstream>
#include <filesystem>
#include<algorithm>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include<Eigen/Core>
#include <Eigen/Dense>
#include <igl/opengl/glfw/Viewer.h>   

using json = nlohmann::json;


//#define _MATERIAL_CURVE_VIEW_DEBUG_


double eval_poly(const Eigen::VectorXd& coeffs, double x) {
    double res = 0;
    for (int i = coeffs.size() - 1; i >= 0; i--) {
        res = res * x + coeffs(i);
    }
    return res;
}

static Eigen::VectorXd polyfit(const Eigen::VectorXd& x_vals, const Eigen::VectorXd& y_vals, int order) {
    int n = x_vals.size();
    Eigen::MatrixXd A(n, order + 1);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < order + 1; j++) {
            A(i, j) = std::pow(x_vals(i), j);
        }
    }
    Eigen::VectorXd coeffs = A.householderQr().solve(y_vals);
    return coeffs;
}


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


void plot_curves(igl::opengl::glfw::Viewer& viewer, const Eigen::VectorXd& coefs,
    double t_min, double t_max, int samples = 200) {

    
    Eigen::MatrixXd V1(samples, 2), V2(samples, 2);

    double div = (t_max - t_min) / samples;
    for (int i = 0; i < samples; ++i) {

        double t1 = t_min + div * i;
        double t2 = t_min + div * (i + 1);

        V1(i, 0) = t1;
        V1(i, 1) = eval_poly(coefs, t1);
        //V1(i, 2) = 0;

        V2(i, 0) = t2;
        V2(i, 1) = eval_poly(coefs, t2);
        //V2(i, 2) = 0;
    }

    Eigen::MatrixXd C(samples, 3);
    C.setZero();
    viewer.data().add_edges(V1, V2, C);

}

void plot_points(igl::opengl::glfw::Viewer& viewer, const std::vector<double>& x_vals,
    const std::vector<double>& y_vals) 
{
    int samples = x_vals.size();
    Eigen::MatrixXd V(samples, 2);

    for (int i = 0; i < samples; ++i) {
        V(i, 0) = x_vals[i];
        V(i, 1) = y_vals[i];
        //V(i, 2) = 0;
    }

    Eigen::MatrixXd C(samples, 3);
    C.setZero();
	C.col(0).setOnes();
    viewer.data().add_points(V, C);

}


void Grayscale_Material::ComputeMaterialCurve()
{
    if (count <= 1)
        return;

	Eigen::VectorXd t_vals_eigen = Eigen::Map<Eigen::VectorXd>(t_vals.data(), t_vals.size());
	Eigen::VectorXd strech_ratio_eigen = Eigen::Map<Eigen::VectorXd>(strech_ratio.data(), strech_ratio.size());
	Eigen::VectorXd youngs_modulus_eigen = Eigen::Map<Eigen::VectorXd>(youngs_modulus.data(), youngs_modulus.size());


	int order_strain_curve = 4;
	int order_modulus_curve = 4;
	Eigen::VectorXd poly_coef_lam_eigen = polyfit(t_vals_eigen, strech_ratio_eigen, order_strain_curve);
	Eigen::VectorXd poly_coef_mod_eigen = polyfit(t_vals_eigen, youngs_modulus_eigen, order_modulus_curve);

	m_strain_curve.order = order_strain_curve + 1;
	m_strain_curve.coeffs = std::vector<double>(poly_coef_lam_eigen.data(), poly_coef_lam_eigen.data() + poly_coef_lam_eigen.size());
	m_moduls_curve.order = order_modulus_curve + 1;
	m_moduls_curve.coeffs = std::vector<double>(poly_coef_mod_eigen.data(), poly_coef_mod_eigen.data() + poly_coef_mod_eigen.size());
    

	std::cout<< "Material curve polynomial coefficients (stretch ratio): \n";
	std::cout << poly_coef_lam_eigen.transpose() << std::endl;
	std::cout << "Material curve polynomial coefficients (Young's modulus): \n";
	std::cout << poly_coef_mod_eigen.transpose() << std::endl;

#ifdef _MATERIAL_CURVE_VIEW_DEBUG_

    igl::opengl::glfw::Viewer material_curve_viewer;
    plot_curves(material_curve_viewer,poly_coef_lam_eigen, 0.0, 1.0);
    plot_points(material_curve_viewer,t_vals, strech_ratio);

    material_curve_viewer.core().is_animating = true;
    material_curve_viewer.core().orthographic = true;
    material_curve_viewer.core().trackball_angle = Eigen::Quaternionf::Identity();
    material_curve_viewer.data().point_size = 5;
	material_curve_viewer.data().line_width = 2;
    material_curve_viewer.launch();

#endif




}




ActiveComposite::ActiveComposite(const std::string& filePath):Grayscale_Material(filePath)
{
    double strain_min = *std::min_element(strech_ratio.begin(), strech_ratio.end());
    double strain_max = *std::max_element(strech_ratio.begin(), strech_ratio.end());

    range_lam = double2{ 1 + strain_min,1 + strain_max };

    double _kappa_ = 1.5 * (strain_max - strain_min) / thickness;

    range_kap = double2{ -_kappa_, _kappa_};
}



