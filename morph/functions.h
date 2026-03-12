#pragma once

#include <TinyAD/Support/GeometryCentral.hh>
#include <TinyAD/ScalarFunction.hh>
#include <geometrycentral/surface/intrinsic_geometry_interface.h>

TinyAD::ScalarFunction<3, double, geometrycentral::surface::VertexRangeF::Etype>
simulationFunction(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                   const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                   const geometrycentral::surface::FaceData<double> &lambda,
                   const geometrycentral::surface::FaceData<double> &kappa,
                   double E,
                   double nu,
                   double h,
                   double w_s,
                   double w_b);

TinyAD::ScalarFunction<3, double, geometrycentral::surface::VertexRangeF::Etype>
simulationFunction(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                   const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                   const geometrycentral::surface::FaceData<double> &lambda,
                   const geometrycentral::surface::VertexData<double> &kappa,
                   double E,
                   double nu,
                   double h,
                   double w_s,
                   double w_b);

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
                              double w_b);

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
                               const geometrycentral::surface::VertexData<double> &kappa,
                               double E,
                               double nu,
                               double h,
                               double w_s,
                               double w_b);

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

// Joint (lambda, kappa) penalty functions using paired feasible values.
// Instead of penalizing lambda and kappa independently, these use 2D distance
// to the paired feasible set, so the optimized variable is steered toward values
// that are jointly feasible with the current fixed parameter.

// For OptKap stage: variable=kappa (per-vertex), fixed=lambda (per-face).
// Computes per-vertex average lambda from adjacent faces, then penalizes
// 2D distance to paired (lam_j, kap_j) feasible points.
TinyAD::ScalarFunction<1, double, Eigen::Index>
JointMaterialPenaltyPerV_OptKap(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                const Eigen::MatrixXi &F,
                                const geometrycentral::surface::FaceData<double> &lambda_pf,
                                const std::vector<double> &feasible_lamb,
                                const std::vector<double> &feasible_kapp,
                                double beta);

// For OptLam stage: variable=lambda (per-face), fixed=kappa (per-vertex).
// Computes per-face average kappa from 3 face vertices, then penalizes
// 2D distance to paired (lam_j, kap_j) feasible points.
TinyAD::ScalarFunction<1, double, Eigen::Index>
JointMaterialPenaltyPerF_OptLam(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                const Eigen::MatrixXi &F,
                                const geometrycentral::surface::VertexData<double> &kappa_pv,
                                const std::vector<double> &feasible_lamb,
                                const std::vector<double> &feasible_kapp,
                                double beta);