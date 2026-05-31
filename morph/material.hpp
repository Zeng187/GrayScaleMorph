#pragma once

#include <string>
#include <vector>
#include <cmath>
#include "common.hpp"
#include<Eigen/core>
#include <tkspline/spline.h>


// Backed by natural cubic spline (tk::spline). Name kept for minimum churn —
// it's now an interpolating spline, not a polynomial.
struct M_Poly_Curve
{
	tk::spline curve;
};

inline double eval_poly(const M_Poly_Curve& c, double t) {
	return c.curve(t);
}

inline double compute_lamb_s(const M_Poly_Curve& c, double t) {
	return 1.0 + eval_poly(c, t);
}

// Bilayer modulus-weighted effective stretch.  (E_i = single-layer modulus,
// s_i = single-layer free strain.)  Reduces to the equal-modulus average
// 1 + 0.5*(s1+s2) when E_1 = E_2.
inline double compute_lamb_d(const M_Poly_Curve& strain_curve,
                             const M_Poly_Curve& modulus_curve,
                             double t1, double t2) {
	const double s1 = eval_poly(strain_curve, t1);
	const double s2 = eval_poly(strain_curve, t2);
	const double E1 = eval_poly(modulus_curve, t1);
	const double E2 = eval_poly(modulus_curve, t2);
	return 1.0 + (E1 * s1 + E2 * s2) / (E1 + E2);
}

inline double compute_modu_s(const M_Poly_Curve& c, double t) {
	return eval_poly(c, t);
}

inline double compute_modu_d(const M_Poly_Curve& c, double t1, double t2) {
	return 0.5 * (eval_poly(c, t1) + eval_poly(c, t2));
}

// Curvature from dual-layer dose pair.
//   Scheme A (kappa_factor != 0): experimentally-calibrated, modulus-INDEPENDENT
//     kappa = kappa_factor * (s1 - s2) / thickness.
//   Fallback (kappa_factor == 0): modulus-weighted bilayer physics, which
//     reduces to 1.5*(s1-s2)/h when E_1=E_2.
inline double compute_curv_d(const M_Poly_Curve& strain_curve,
                             const M_Poly_Curve& modulus_curve,
                             double thickness, double t1, double t2,
                             double kappa_factor = 0.0) {
	const double s1 = eval_poly(strain_curve, t1);
	const double s2 = eval_poly(strain_curve, t2);
	if (kappa_factor != 0.0) {
		// Scheme A: experimentally-calibrated, modulus-independent.
		return kappa_factor * (s1 - s2) / thickness;
	}
	// Fallback: modulus-weighted bilayer physics.
	const double E1 = eval_poly(modulus_curve, t1);
	const double E2 = eval_poly(modulus_curve, t2);
	const double S  = E1 + E2;
	const double P  = E1 * E2;
	return 24.0 * P * (s1 - s2) / (thickness * (S * S + 12.0 * P));
}

// Thin-plate spline (TPS) representation of E_eff(lambda, kappa^2).
// Off-line construction (see S0_MaterialGen/material_gen_tps.py):
//   1. Compute N=28 unique anchor (lambda_k, kappa_k, E_eff_k) via the
//      modulus-weighted bilayer formulas above on the 7x7 dose grid
//      (i <= j to drop the kappa-sign duplicates).
//   2. Normalize: lambda_hat = (lambda - lambda_mid) / lambda_half_range,
//                 kappa_sq_hat = kappa^2 / kappa_sq_max.
//   3. Solve [K P; P^T 0] [a; b] = [E; 0] for the N+3 weights, where
//      K_{ij} = phi(||p_i - p_j||), P_{i,:} = [1, lambda_hat_i, kappa_sq_hat_i],
//      phi(r) = r^2 * log(r).
struct M_Surface_TPS {
    std::vector<double> anchors_lambda_hat;     // size N
    std::vector<double> anchors_kappa_sq_hat;   // size N
    std::vector<double> tps_weights;            // size N + 3 (a_1..a_N, b_0, b_1, b_2)
    double lambda_mid = 0.0;
    double lambda_half_range = 1.0;
    double kappa_sq_max = 1.0;
    bool loaded = false;
};

// Backwards-compatible alias: existing call-sites in functions.cpp/h and
// newton.cpp/h still spell out M_Surface_LK; they now resolve to the TPS
// struct without touching their signatures.
using M_Surface_LK = M_Surface_TPS;

// TinyAD-friendly TPS evaluation.  T may be double or a TinyAD scalar.
//   phi(r) = r^2 log(r) = 0.5 * r^2 * log(r^2), with phi(0) := 0.
//
// Soft floor at E_min = 0.1 MPa: defensive against TPS extrapolation outside
// the convex hull of the 28 anchors going non-physical.  Smooth approximation
// of max(E, E_min) so TinyAD gradients stay defined everywhere.
template <typename T>
inline T compute_E_lk(const M_Surface_TPS& s, T lambda, T kappa) {
    using std::log;
    using std::sqrt;

    const T lh  = (lambda - T(s.lambda_mid)) / T(s.lambda_half_range);
    const T ksh = (kappa * kappa) / T(s.kappa_sq_max);

    const int N = static_cast<int>(s.anchors_lambda_hat.size());
    T E = T(0);

    // TPS radial basis sum
    for (int k = 0; k < N; ++k) {
        const T dx = lh  - T(s.anchors_lambda_hat[k]);
        const T dy = ksh - T(s.anchors_kappa_sq_hat[k]);
        const T r2 = dx * dx + dy * dy;
        // phi(r) = 0.5 * r^2 * log(r^2);  phi(0) := 0.
        const T basis = (r2 > T(1e-30)) ? T(0.5) * r2 * log(r2) : T(0);
        E += T(s.tps_weights[k]) * basis;
    }

    // Affine part: b_0 + b_1*lambda_hat + b_2*kappa_sq_hat
    E += T(s.tps_weights[N    ]);
    E += T(s.tps_weights[N + 1]) * lh;
    E += T(s.tps_weights[N + 2]) * ksh;

    // Smooth floor: max(E, E_min) with C^infty approximation.
    const T E_min = T(0.1);
    const T diff = E - E_min;
    return E_min + T(0.5) * (diff + sqrt(diff * diff + T(1e-4)));
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
	double kappa_factor = 0.0;   // scheme-A curvature coefficient; 0 => use physics fallback


	M_Poly_Curve m_strain_curve;
	M_Poly_Curve m_moduls_curve;

};

class ActiveComposite: public Grayscale_Material
{
public:
	ActiveComposite(const std::string& filePath);

	void ComputeFeasibleVals();
	void LoadEsurface(const std::string& filePath);

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

	M_Surface_TPS m_E_surface;

};

