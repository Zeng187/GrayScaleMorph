#pragma once

#include <Eigen/Core>
#include <TinyAD/Support/GeometryCentral.hh>
#include <TinyAD/ScalarFunction.hh>
#include <geometrycentral/surface/intrinsic_geometry_interface.h>

#include <functional>
#include <ostream>
#include <string>

struct M_Poly_Curve;

/// Per-iteration log handle for the SGN _Penalty solvers.  When `csv` is
/// non-null, the solver appends one CSV row at the initial state (iter=-1)
/// and one row after each Newton step (iter=0..max_iters-1) with all
/// objective components.  `stage`, `phase`, and `beta` are written as tag
/// columns so multiple stages of the outer alternating loop share one CSV.
struct InnerIterLog {
    std::ostream* csv  = nullptr;
    int           stage = -1;
    std::string   phase;       ///< "OptKap" or "OptLam"
    double        beta  = 0.0; ///< softmin sharpness from the outer cfg
};

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
const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
const std::vector<int>& fixedIdx,
int max_iters,
double lim,
double wM,
double wL,
const geometrycentral::surface::FaceData<double>& E_face,
double nu,
double h,
double w_s,
double w_b,
const std::vector<int>& ref_faces,
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {});



Eigen::MatrixXd sparse_gauss_newton_FixKap_OptLam(
geometrycentral::surface::IntrinsicGeometryInterface& geometry,
const Eigen::MatrixXd& targetV,
const Eigen::MatrixXd& initV,
const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
geometrycentral::surface::FaceData<double>& theta1,
geometrycentral::surface::FaceData<double>& theta2,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
const std::vector<int>& fixedIdx,
int max_iters,
double lim,
double wM,
double wL,
const geometrycentral::surface::FaceData<double>& E_face,
double nu,
double h,
double w_s,
double w_b,
const std::vector<int>& ref_faces,
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {});





Eigen::MatrixXd sparse_gauss_newton_FixLam_OptKap_Penalty(
geometrycentral::surface::IntrinsicGeometryInterface& geometry,
const Eigen::MatrixXd& targetV,
const Eigen::MatrixXd& initV,
const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
geometrycentral::surface::FaceData<double>& theta1,
geometrycentral::surface::FaceData<double>& theta2,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& penaltyFunc,
const std::vector<int>& fixedIdx,
int max_iters,
double lim,
double wM,
double wL,
double wP,
const geometrycentral::surface::FaceData<double>& E_face,
double nu,
double h,
double w_s,
double w_b,
const std::vector<int>& ref_faces,
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {},
const InnerIterLog& iter_log = {});




Eigen::MatrixXd sparse_gauss_newton_FixKap_OptLam_Penalty(
geometrycentral::surface::IntrinsicGeometryInterface& geometry,
const Eigen::MatrixXd& targetV,
const Eigen::MatrixXd& initV,
const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
geometrycentral::surface::FaceData<double>& theta1,
geometrycentral::surface::FaceData<double>& theta2,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
const TinyAD::ScalarFunction<1, double, Eigen::Index>& penaltyFunc,
const std::vector<int>& fixedIdx,
int max_iters,
double lim,
double wM,
double wL,
double wP,
const geometrycentral::surface::FaceData<double>& E_face,
double nu,
double h,
double w_s,
double w_b,
const std::vector<int>& ref_faces,
const std::function<void(const Eigen::VectorXd&)>& callback = [](const auto&) {},
const InnerIterLog& iter_log = {});
