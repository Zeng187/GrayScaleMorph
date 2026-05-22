#pragma once

#include <Eigen/Core>
#include <TinyAD/Support/GeometryCentral.hh>
#include <TinyAD/ScalarFunction.hh>
#include <geometrycentral/surface/intrinsic_geometry_interface.h>

#include <functional>

// Per-SGN-iter metrics callback.  Called at the end of each Newton iteration
// inside *_MGDA and the non-penalty / penalty SGN functions so the host
// (main.cpp) can stream a row into a CSV / metrics file.  Mirrors the
// signature used by S2_GrayScaleMorph's iter_logger for cross-method CSV
// compatibility.
//   iter      : 0-based iter index inside the SGN call
//   x_iter    : flat 3*nV vector of the SGN's current vertex positions,
//               so the caller can compute proj_dist at this exact state
//   spn       : SPN energy = distance + self_reg + other_reg [+ wP*phi]
//   distance  : mass-weighted distance to target
//   self_reg  : regulariser of the variable currently being optimised
//   penalty   : feasibility penalty value (= wP*phi for penalty variants,
//               = phi for MGDA variants, = 0 for non-penalty SGN)
using SgnIterCallback = std::function<void(int iter,
                                           const Eigen::VectorXd& x_iter,
                                           double spn,
                                           double distance,
                                           double self_reg,
                                           double penalty)>;

inline const SgnIterCallback sgn_iter_noop = [](int, const Eigen::VectorXd&, double, double, double, double){};

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
const std::vector<double>& candidate_vals,    // 1D candidates for the optimised variable
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
const SgnIterCallback& iter_cb = sgn_iter_noop,
// Optional 2D-joint penalty inputs.  When cand_other is empty (default),
// d_P uses 1D snap on candidate_vals.  When cand_other is non-empty,
// d_P uses 2D Euclidean joint snap: pick j* = arg min_j [(theta_self -
// candidate_vals[j])^2 + (other_const_f - cand_other[j])^2], snap target
// = candidate_vals[j*].  other_const must have length == nF.
const std::vector<double>& cand_other  = std::vector<double>{},
const std::vector<double>& other_const = std::vector<double>{},
// Override the closed-form MGDA alpha (combination weight for d_F vs d_P).
// Negative -> use the standard mgda_alpha(d_F, d_P).  Non-negative in
// [0, 1] -> bypass the closed-form and use this value directly so the
// caller can impose a stage-wise alpha schedule.
double alpha_override = -1.0);


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
const std::vector<double>& candidate_vals,    // 1D candidates for the optimised variable
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
const SgnIterCallback& iter_cb = sgn_iter_noop,
// See FixLam_OptKap_MGDA above; same semantics.  Pass cand_other &
// other_const non-empty to switch d_P to 2D Euclidean joint snap.
const std::vector<double>& cand_other  = std::vector<double>{},
const std::vector<double>& other_const = std::vector<double>{},
double alpha_override = -1.0);


// ---------------------------------------------------------------------------
// SGN OptP: optimise the 2D parameterisation P with material (lambda,
// kappa) held constant.  Variables are [x (3|V|), P (2|V|)].  At each SGN
// iter:
//   1. P updated -> MrInv recomputed -> forward Newton solves x to equilibrium.
//   2. KKT system gives joint (Δx, ΔP) step minimising
//        distance(x*(P), x_T) + wM_P * ||P - P_anchor||^2
//                              + wL_P * P^T L_P P + other_reg
//      under the forward equilibrium constraint.
// `P_io` is updated in place to the final P.  Returns the updated Vr
// (equilibrium V at the final P).  Replaces the lambda-aware ARAP P-update.
Eigen::MatrixXd sparse_gauss_newton_FixMaterial_OptP(
geometrycentral::surface::IntrinsicGeometryInterface& geometry,
const Eigen::MatrixXi& F,
const Eigen::MatrixXd& targetV,
const Eigen::MatrixXd& initV,
Eigen::MatrixXd& P_io,
const geometrycentral::surface::FaceData<double>& lambda_pf,
const geometrycentral::surface::FaceData<double>& kappa_pf,
const Eigen::VectorXd& masses,
const Eigen::SparseMatrix<double>& M_P,
const Eigen::SparseMatrix<double>& L_P,
const Eigen::MatrixXd& P_anchor,
double other_reg,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
const std::vector<int>& fixedIdx,
int max_iters,
double lim,
double wM_P,
double wL_P,
double E,
double nu,
double h,
double w_s,
double w_b,
const std::vector<int>& ref_faces,
double& final_distance,
double& final_spn_energy,
double& final_self_reg,
const SgnIterCallback& iter_cb = sgn_iter_noop,
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {});
