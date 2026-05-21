#pragma once

#include <TinyAD/Support/GeometryCentral.hh>
#include <TinyAD/ScalarFunction.hh>
#include <geometrycentral/surface/intrinsic_geometry_interface.h>

#include <vector>

TinyAD::ScalarFunction<3, double, geometrycentral::surface::VertexRangeF::Etype>
simulationFunction(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                   const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                   const geometrycentral::surface::FaceData<double> &lambda,
                   const geometrycentral::surface::FaceData<double> &kappa,
                   double E,
                   double nu,
                   double h,
                   double w_s,
                   double w_b,
                   const std::vector<int> &ref_faces);

TinyAD::ScalarFunction<3, double, geometrycentral::surface::VertexRangeF::Etype>
simulationFunction(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                   const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                   const geometrycentral::surface::FaceData<double> &lambda,
                   const geometrycentral::surface::VertexData<double> &kappa,
                   double E,
                   double nu,
                   double h,
                   double w_s,
                   double w_b,
                   const std::vector<int> &ref_faces);

TinyAD::ScalarFunction<3, double, geometrycentral::surface::VertexRangeF::Etype>
simulationFunction(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                   const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                   const geometrycentral::surface::VertexData<double> &lambda,
                   const geometrycentral::surface::VertexData<double> &kappa,
                   double E,
                   double nu,
                   double h,
                   double w_s,
                   double w_b);

// Forward declaration for material curve
struct M_Poly_Curve;

// Simulation function with material-based lambda/kappa computation
// Computes lambda and kappa from t_layer_1, t_layer_2 vertex data using material curves
TinyAD::ScalarFunction<3, double, geometrycentral::surface::VertexRangeF::Etype>
simulationFunctionWithMaterial(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                               const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                               const geometrycentral::surface::VertexData<double> &t_layer_1,
                               const geometrycentral::surface::VertexData<double> &t_layer_2,
                               const M_Poly_Curve &strain_curve,
                               const M_Poly_Curve &moduls_curve,
                               double E,
                               double nu,
                               double h,
                               double w_s,
                               double w_b);
                               
TinyAD::ScalarFunction<3, double, geometrycentral::surface::VertexRangeF::Etype>
simulationFunctionWithMaterial(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                               const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                               const geometrycentral::surface::FaceData<double> &t_layer_1,
                               const geometrycentral::surface::FaceData<double> &t_layer_2,
                               const M_Poly_Curve &strain_curve,
                               const M_Poly_Curve &moduls_curve,
                               double E,
                               double nu,
                               double h,
                               double w_s,
                               double w_b);

TinyAD::ScalarFunction<1, double, Eigen::Index>
adjointFunction_FixLam_OptKap(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                              const Eigen::MatrixXi &F,
                              const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                              const geometrycentral::surface::FaceData<double> &lambda,
                              double E,
                              double nu,
                              double h,
                              double w_s,
                              double w_b,
                              const std::vector<int> &ref_faces);

TinyAD::ScalarFunction<1, double, Eigen::Index>
adjointFunction_FixLam_OptKap(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                              const Eigen::MatrixXi &F,
                              const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                              const geometrycentral::surface::VertexData<double> &lambda,
                              double E,
                              double nu,
                              double h,
                              double w_s,
                              double w_b);

TinyAD::ScalarFunction<1, double, Eigen::Index>
adjointFunction_FixKap_OptLam(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                              const Eigen::MatrixXi &F,
                              const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                              const geometrycentral::surface::VertexData<double> &kappa,
                              double E,
                              double nu,
                              double h,
                              double w_s,
                              double w_b);

TinyAD::ScalarFunction<1, double, Eigen::Index>
adjointFunction_FixKap_OptLam2(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                               const Eigen::MatrixXi &F,
                               const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                               const geometrycentral::surface::FaceData<double> &kappa,
                               double E,
                               double nu,
                               double h,
                               double w_s,
                               double w_b,
                               const std::vector<int> &ref_faces);

TinyAD::ScalarFunction<1, double, Eigen::Index>
adjointFunctionWithMaterial_Lay1(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                 const Eigen::MatrixXi &F,
                                 const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                                 const geometrycentral::surface::VertexData<double> &t_layer_1,
                                 const M_Poly_Curve &lambda_curve,
                                 const M_Poly_Curve &kappa_curve,
                                 double E,
                                 double nu,
                                 double h,
                                 double w_s,
                                 double w_b);

TinyAD::ScalarFunction<1, double, Eigen::Index>
adjointFunctionWithMaterial_Lay2(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                 const Eigen::MatrixXi &F,
                                 const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                                 const geometrycentral::surface::VertexData<double> &t_layer_2,
                                 const M_Poly_Curve &lambda_curve,
                                 const M_Poly_Curve &kappa_curve,
                                 double E,
                                 double nu,
                                 double h,
                                 double w_s,
                                 double w_b);

// Material penalty function to encourage values towards feasible material property values
// Creates a smooth penalty that pushes vertex values toward the provided feasible_vals
// beta controls the sharpness of the penalty (higher = sharper)
TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialPenaltyFunctionPerV(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                            const std::vector<double> &feasible_vals,
                            double beta);

TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialPenaltyFunctionPerF(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                            const std::vector<double> &feasible_vals,
                            double beta);

// -- 2D joint hard-min penalties --------------------------------------------
//
// `feasible_kapp[i]` and `feasible_lamb[i]` together describe the i-th
// feasible (t1, t2) pair.  These joint penalties take the min over i of the
// 2D distance to the candidate pair, so the argmin agrees with
// `find_feasible_idx` exactly: the penalty gradient never pulls toward a
// "virtual" pair (kappa-from-pair-A, lambda-from-pair-B) that doesn't exist
// in the feasible set.
//
// Distance to pair i:   (kappa - feasible_kapp[i])^2 + alpha * (lambda - feasible_lamb[i])^2
//
// `alpha` re-weights the lambda contribution because kappa and lambda have
// different numerical magnitudes.  Argmin is done via value-only comparison
// inside the TinyAD lambda, so AD only tracks the surviving (variable - cand)^2
// branch and the gradient strictly points toward the joint-nearest pair.
//
// _OptKap variant: variable = kappa, lambda is held constant from `lambda_pf`.
// _OptLam variant: variable = lambda, kappa  is held constant from `kappa_pf`.
TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialJointPenaltyPerF_OptKap(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                const geometrycentral::surface::FaceData<double> &lambda_pf,
                                const std::vector<double> &feasible_kapp,
                                const std::vector<double> &feasible_lamb,
                                double alpha,
                                double beta);

TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialJointPenaltyPerF_OptLam(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                const geometrycentral::surface::FaceData<double> &kappa_pf,
                                const std::vector<double> &feasible_kapp,
                                const std::vector<double> &feasible_lamb,
                                double alpha,
                                double beta);