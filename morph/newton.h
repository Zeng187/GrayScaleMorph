#pragma once

#include <Eigen/Core>
#include <TinyAD/Support/GeometryCentral.hh>
#include <TinyAD/ScalarFunction.hh>
#include <geometrycentral/surface/intrinsic_geometry_interface.h>

#include <functional>

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
double theta_anchor,
double E,
double nu,
double h,
double w_s,
double w_b,
const std::vector<int>& ref_faces,
double& final_distance,
double& final_spn_energy,
double& final_self_reg,
const std::function<void(int, const Eigen::VectorXd&, double, double, double, double)>& iter_logger
    = [](int, const Eigen::VectorXd&, double, double, double, double){},
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {});


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
double theta_anchor,
double E,
double nu,
double h,
double w_s,
double w_b,
const std::vector<int>& ref_faces,
double& final_distance,
double& final_spn_energy,
double& final_self_reg,
const std::function<void(int, const Eigen::VectorXd&, double, double, double, double)>& iter_logger
    = [](int, const Eigen::VectorXd&, double, double, double, double){},
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {});


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
double theta_anchor,
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
const std::function<void(int, const Eigen::VectorXd&, double, double, double, double)>& iter_logger
    = [](int, const Eigen::VectorXd&, double, double, double, double){},
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
double theta_anchor,
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
const std::function<void(int, const Eigen::VectorXd&, double, double, double, double)>& iter_logger
    = [](int, const Eigen::VectorXd&, double, double, double, double){},
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {});


// SGN for OptP: optimise the 2D parameterisation P with material (lambda,
// kappa) held constant.  Variables are [x (3|V|), P (2|V|)].  At each SGN
// iter:
//   1. P updated -> MrInv recomputed -> forward Newton solves x to equilibrium.
//   2. KKT system gives joint (Δx, ΔP) step minimising the SPN target
//      distance(x*(P), x_T) + wM_P * ||P - P_anchor||^2 + wL_P * ||grad P||^2
//      + other_reg, while enforcing the forward equilibrium constraint.
//
// Returns the updated Vr (equilibrium V at the final P).  `P_io` is updated
// in place to the final P.
Eigen::MatrixXd sparse_gauss_newton_FixMaterial_OptP(
geometrycentral::surface::IntrinsicGeometryInterface& geometry,
const Eigen::MatrixXi& F,
const Eigen::MatrixXd& targetV,
const Eigen::MatrixXd& initV,
Eigen::MatrixXd& P_io,                                                 // 2D param, in/out
const geometrycentral::surface::FaceData<double>& lambda_pf,
const geometrycentral::surface::FaceData<double>& kappa_pf,
const Eigen::VectorXd& masses,                                         // 3|V|, x mass
const Eigen::SparseMatrix<double>& M_P,                                // 2|V|x2|V|, P mass (diag vertex area * I_2)
const Eigen::SparseMatrix<double>& L_P,                                // 2|V|x2|V|, P Laplacian (cotan * I_2)
const Eigen::MatrixXd& P_anchor,                                       // |V|x2, reference P for anchor reg
double other_reg,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,    // FixMaterial_OptP version
const std::vector<int>& fixedIdx,                                      // fixed x DOFs (size 9 = 3 verts x 3)
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
const std::function<void(int, const Eigen::VectorXd&, double, double, double, double)>& iter_logger
    = [](int, const Eigen::VectorXd&, double, double, double, double){},
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {});
