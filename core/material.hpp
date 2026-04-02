#pragma once

#include <string>
#include <vector>
#include <cmath>
#include "common.hpp"
#include<Eigen/Core>


struct M_Poly_Curve
{
	std::vector<double> coeffs;
	int order = 0;
};

template <typename T>
inline T eval_poly(const M_Poly_Curve& _curve, T x) {
	T res = T(0);
	for (int i = _curve.order - 1; i >= 0; i--) {
		res = res * x + T(_curve.coeffs[i]);
	}
	return res;
}

template <typename T>
inline T compute_lamb_s(const M_Poly_Curve& _curve, T t) {
	return 1 + eval_poly(_curve, t);
}

template <typename T>
inline T compute_lamb_d(const M_Poly_Curve& _curve, T t1, T t2) {
	T val_1 = eval_poly(_curve, t1);
	T val_2 = eval_poly(_curve, t2);
	return 1 + T(0.5) * (val_1 + val_2);
}

template <typename T>
inline T compute_modu_s(const M_Poly_Curve& _curve, T t) {
	return eval_poly(_curve, t);
}

template <typename T>
inline T compute_modu_d(const M_Poly_Curve& _curve, T t1, T t2) {
	T val_1 = eval_poly(_curve, t1);
	T val_2 = eval_poly(_curve, t2);
	return T(0.5) * (val_1 + val_2);
}

template <typename T>
inline T compute_curv_d(const M_Poly_Curve& _curve, double thickness, T t1, T t2) {
	T val_1 = eval_poly(_curve, t1);
	T val_2 = eval_poly(_curve, t2);
	return T(1.5) * (val_2 - val_1) / T(thickness);
}

// Find the index of the nearest feasible (kap, lam) pair to (kap, lam) jointly
inline int find_feasible_idx(const std::vector<double>& feas_kap,
                              const std::vector<double>& feas_lam,
                              double kap, double lam)
{
    int best_i = 0;
    double best_d = std::pow(kap - feas_kap[0], 2) + std::pow(lam - feas_lam[0], 2);
    for (size_t i = 1; i < feas_kap.size(); ++i) {
        double d = std::pow(kap - feas_kap[i], 2) + std::pow(lam - feas_lam[i], 2);
        if (d < best_d) { best_d = d; best_i = static_cast<int>(i); }
    }
    return best_i;
}

// Find the nearest feasible value to t in feas_vals
inline double project_to_feasible(const std::vector<double>& feas_vals, double t)
{
    if (feas_vals.empty()) return t;
    double best = feas_vals[0];
    double best_d = std::abs(t - best);
    for (size_t i = 1; i < feas_vals.size(); ++i) {
        double d = std::abs(t - feas_vals[i]);
        if (d < best_d) { best_d = d; best = feas_vals[i]; }
    }
    return best;
}

// Compute the absolute difference between t and its closest value in feas_vals
inline double compute_candidate_diff(const std::vector<double> &feas_vals, double t)
{
	if (feas_vals.empty()) {
		return 0.0;
	}

	double min_diff = std::abs(t - feas_vals[0]);
	for (size_t i = 1; i < feas_vals.size(); ++i) {
		double diff = std::abs(t - feas_vals[i]);
		if (diff < min_diff) {
			min_diff = diff;
		}
	}

	return min_diff;
}

// Compute the maximal or average distance from vals to closest points in feas_vals
// Returns the maximum difference by default
inline double compute_candidate_diff(const std::vector<double> &feas_vals, const Eigen::VectorXd& vals, bool use_max = true)
{
	if (vals.size() == 0 || feas_vals.empty()) {
		return 0.0;
	}

	double total_diff = 0.0;
	double max_diff = 0.0;

	for (int i = 0; i < vals.size(); ++i) {
		double diff = compute_candidate_diff(feas_vals, vals[i]);
		total_diff += diff;
		if (diff > max_diff) {
			max_diff = diff;
		}
	}

	if (use_max) {
		return max_diff;
	} else {
		return total_diff / vals.size();
	}
}

// Invert polynomial curve: given strain value, find t parameter using bisection
inline double invert_poly(const M_Poly_Curve& _curve, double target_strain, double t_min = 0.0, double t_max = 1.0, double tol = 1e-6, int max_iter = 100) {
	// Use bisection method to find t such that eval_poly(curve, t) = target_strain
	double a = t_min;
	double b = t_max;
	double fa = eval_poly(_curve, a) - target_strain;
	double fb = eval_poly(_curve, b) - target_strain;

	// If target is outside the range, clamp to boundaries
	if (fa * fb > 0) {
		// Same sign, target might be outside range
		if (std::abs(fa) < std::abs(fb)) {
			return a;
		} else {
			return b;
		}
	}

	// Bisection
	for (int i = 0; i < max_iter; ++i) {
		double c = (a + b) / 2.0;
		double fc = eval_poly(_curve, c) - target_strain;

		if (std::abs(fc) < tol || (b - a) / 2.0 < tol) {
			return c;
		}

		if (fa * fc < 0) {
			b = c;
			fb = fc;
		} else {
			a = c;
			fa = fc;
		}
	}

	return (a + b) / 2.0;
}

class Grayscale_Material
{
public:
	Grayscale_Material(const std::string& filePath);
	~Grayscale_Material(){};

	void ComputeMaterialCurve();

	std::string name="";
	std::string description = "";
	
	std::vector<double> t_vals;
	std::vector<double> youngs_modulus;
	std::vector<double> strech_ratio;
	int count =0;

	double thickness = 1.0;


	M_Poly_Curve m_strain_curve;
	M_Poly_Curve m_moduls_curve;

	bool curves_loaded = false;
	std::string material_file_path;
};

class ActiveComposite: public Grayscale_Material
{
public:
	ActiveComposite(const std::string& filePath);

	void ComputeFeasibleVals();

	// std::vector<double> lambda;
	// std::vector<double> kappa;
	// std::vector<double> E_moduls;
	double2 range_lam;
	double2 range_kap;

	int fesasible_cnt = 0;
	std::vector<std::pair<double,double>> feasible_t_vals;
	std::vector<double> feasible_lamb;
	std::vector<double> feasible_kapp;
	std::vector<double> feasible_modl;
	// M_Poly_Curve m_lambda_curve;
	// M_Poly_Curve m_kappa_curve;
	// M_Poly_Curve m_moduls_curve;


};

