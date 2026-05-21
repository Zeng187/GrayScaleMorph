#pragma once

#include <Eigen/Core>
#include <TinyAD/Support/GeometryCentral.hh>
#include <TinyAD/ScalarFunction.hh>
#include <geometrycentral/surface/intrinsic_geometry_interface.h>

#include <functional>

// Per-SGN-iter metrics callback.  Called at the end of each Newton iteration
// inside the *_MGDA and non-penalty SGN functions so the host (main.cpp)
// can stream a row into a CSV / metrics file.
//   iter      : 0-based iter index inside the SGN call
//   F         : SPN energy (distance + self_reg + other_reg [+ wP*phi for non-MGDA])
//   distance  : mass-weighted distance to target
//   phi       : feasibility penalty value (0 for non-penalty SGN)
//   self_reg  : regulariser of the variable currently being optimised
//   other_reg : regulariser of the variable currently held constant
using SgnIterCallback = std::function<void(int iter,
                                           double F,
                                           double distance,
                                           double phi,
                                           double self_reg,
                                           double other_reg)>;

inline const SgnIterCallback sgn_iter_noop = [](int, double, double, double, double, double){};

template <class Func, class Solver>
void newton(
    Eigen::VectorXd& x,
    Func& func,
    Solver& solver,
    int max_iters = 1000,
    double lim = 1e-6,
    bool verbose = true,
    const std::vector<int>& fixedIdx = {},
    const std::function<void(const Eigen::VectorXd&)>& callBack = [](const auto&) {});


template <class Func>
void newton(
    geometrycentral::surface::IntrinsicGeometryInterface& geometry,
    Eigen::MatrixXd& V,
    Func& func,
    int max_iters,
    double lim,
    bool verbose = true,
    const std::vector<int>& fixedIdx = {},
    const std::function<void(const Eigen::VectorXd&)>& callBack = [](const auto&) {});

Eigen::MatrixXd sparse_gauss_newton_FixLam_OptKap(
geometrycentral::surface::IntrinsicGeometryInterface& geometry,
const Eigen::MatrixXd& targetV,
const Eigen::MatrixXd& initV,
const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
geometrycentral::surface::FaceData<double>& theta1,
geometrycentral::surface::FaceData<double>& theta2,
const Eigen::VectorXd& masses,
double other_reg,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
const std::vector<int>& fixedIdx,
int max_iters,
double lim,
double wM,
double wL,
double E,
double nu,
double h,
double w_s,
double w_b,
const std::vector<int>& ref_faces,
double& final_distance,
double& final_spn_energy,
double& final_self_reg,
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {},
const SgnIterCallback& iter_cb = sgn_iter_noop);


Eigen::MatrixXd sparse_gauss_newton_FixKap_OptLam(
geometrycentral::surface::IntrinsicGeometryInterface& geometry,
const Eigen::MatrixXd& targetV,
const Eigen::MatrixXd& initV,
const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
geometrycentral::surface::FaceData<double>& theta1,
geometrycentral::surface::FaceData<double>& theta2,
const Eigen::VectorXd& masses,
double other_reg,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
const std::vector<int>& fixedIdx,
int max_iters,
double lim,
double wM,
double wL,
double E,
double nu,
double h,
double w_s,
double w_b,
const std::vector<int>& ref_faces,
double& final_distance,
double& final_spn_energy,
double& final_self_reg,
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {},
const SgnIterCallback& iter_cb = sgn_iter_noop);


Eigen::MatrixXd sparse_gauss_newton_FixLam_OptKap_Penalty(
geometrycentral::surface::IntrinsicGeometryInterface& geometry,
const Eigen::MatrixXd& targetV,
const Eigen::MatrixXd& initV,
const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
geometrycentral::surface::FaceData<double>& theta1,
geometrycentral::surface::FaceData<double>& theta2,
const Eigen::VectorXd& masses,
double other_reg,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& penaltyFunc,
const std::vector<int>& fixedIdx,
int max_iters,
double lim,
double wM,
double wL,
double wP,
double E,
double nu,
double h,
double w_s,
double w_b,
const std::vector<int>& ref_faces,
double& final_distance,
double& final_spn_energy,
double& final_self_reg,
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {});


Eigen::MatrixXd sparse_gauss_newton_FixKap_OptLam_Penalty(
geometrycentral::surface::IntrinsicGeometryInterface& geometry,
const Eigen::MatrixXd& targetV,
const Eigen::MatrixXd& initV,
const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
geometrycentral::surface::FaceData<double>& theta1,
geometrycentral::surface::FaceData<double>& theta2,
const Eigen::VectorXd& masses,
double other_reg,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& penaltyFunc,
const std::vector<int>& fixedIdx,
int max_iters,
double lim,
double wM,
double wL,
double wP,
double E,
double nu,
double h,
double w_s,
double w_b,
const std::vector<int>& ref_faces,
double& final_distance,
double& final_spn_energy,
double& final_self_reg,
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {});


// ---------------------------------------------------------------------------
// MGDA (Multiple Gradient Descent) variants — see docs/grayscalemorph/inverse_mgda.md
//
// Treats SPN energy F = (distance + kappa_reg + lambda_reg) and feasibility
// penalty Phi as two independent objectives.  Each iter:
//   d_F = -H_F^{-1} * grad F   (from the KKT system, wP=0)
//   d_P = snap(theta) - theta  (closed form: hard-min penalty Hessian = (2*beta/nF)*I)
//   alpha  = mgda_alpha(d_F, d_P)
//   d      = alpha * d_F + (1 - alpha) * d_P
//   step s by two-objective Armijo (lineSearchMulti) on F and Phi
//
// Terminates on Pareto-critical:  ||alpha * gF + (1-alpha) * gP|| < lim,
// or when ||d||^2 < lim, or max_iters.  No wP/wM/wL homotopy.
//
// candidate_vals is the list of feasible projection targets for the optimised
// variable (theta2 here): the hard-min Hessian assumes one nearest candidate
// per face (Voronoi cell), so snap(theta_f) = argmin_C |theta_f - C|.
// betaP scales the penalty so dP magnitude is comparable to dF; pass the
// same betaP value used when constructing the TinyAD penaltyFunc.
// ---------------------------------------------------------------------------

Eigen::MatrixXd sparse_gauss_newton_FixLam_OptKap_MGDA(
geometrycentral::surface::IntrinsicGeometryInterface& geometry,
const Eigen::MatrixXd& targetV,
const Eigen::MatrixXd& initV,
const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
geometrycentral::surface::FaceData<double>& theta1,
geometrycentral::surface::FaceData<double>& theta2,
const Eigen::VectorXd& masses,
double other_reg,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& penaltyFunc,
const std::vector<double>& candidate_vals,
double betaP,
const std::vector<int>& fixedIdx,
int max_iters,
double lim,
double wM,
double wL,
double E,
double nu,
double h,
double w_s,
double w_b,
const std::vector<int>& ref_faces,
double& final_distance,
double& final_spn_energy,
double& final_self_reg,
double& final_penalty,
double& final_pareto_norm,
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {},
const SgnIterCallback& iter_cb = sgn_iter_noop);


Eigen::MatrixXd sparse_gauss_newton_FixKap_OptLam_MGDA(
geometrycentral::surface::IntrinsicGeometryInterface& geometry,
const Eigen::MatrixXd& targetV,
const Eigen::MatrixXd& initV,
const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
geometrycentral::surface::FaceData<double>& theta1,
geometrycentral::surface::FaceData<double>& theta2,
const Eigen::VectorXd& masses,
double other_reg,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& penaltyFunc,
const std::vector<double>& candidate_vals,
double betaP,
const std::vector<int>& fixedIdx,
int max_iters,
double lim,
double wM,
double wL,
double E,
double nu,
double h,
double w_s,
double w_b,
const std::vector<int>& ref_faces,
double& final_distance,
double& final_spn_energy,
double& final_self_reg,
double& final_penalty,
double& final_pareto_norm,
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {},
const SgnIterCallback& iter_cb = sgn_iter_noop);
