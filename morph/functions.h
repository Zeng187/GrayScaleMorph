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

// -- 2D joint hard-min penalty in elastic-energy metric --------------------
//
// `feasible_kapp[i]` and `feasible_lamb[i]` describe the i-th feasible
// (t1, t2) pair.  The joint penalty takes the min over i of the Efrati
// non-Euclidean plate strain-energy distance to that pair:
//
//   d^2_i(lambda, kappa) = lambdabar_i^2 * (E/(1-nu)) *
//                          [ (h/4)   * (lambda^2 - lambdabar_i^2)^2
//                          + (h^3/12)* (kappa  - kappabar_i  )^2 ]
//
//   penalty_per_face = beta / nF * min_i d^2_i
//
// The bracket has the correct thickness exponents (h for stretching, h^3 for
// bending), and the leading lambdabar_i^2 is the candidate-side area element
// sqrt(det(gbar_i)).  Different kappa/lambda units are naturally rebalanced
// by this physical metric, so no manual `alpha` is needed.
//
// argmin is computed via value-only comparison so TinyAD only tracks the
// surviving (variable - cand)^2 branch.  The "other" dimension is a constant
// captured by reference from the per-face FaceData; OptKap holds lambda
// constant from `lambda_pf`, OptLam holds kappa constant from `kappa_pf`.
//
// E (Young's modulus) is fixed at 1.0 in these helpers to stay consistent
// with the SGN forward-sim setup; downstream uses pass `beta` to scale the
// overall penalty weight.
TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialJointPenaltyPerF_OptKap(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                const geometrycentral::surface::FaceData<double> &lambda_pf,
                                const std::vector<double> &feasible_kapp,
                                const std::vector<double> &feasible_lamb,
                                double thickness,
                                double poisson_ratio,
                                double beta);

TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialJointPenaltyPerF_OptLam(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                const geometrycentral::surface::FaceData<double> &kappa_pf,
                                const std::vector<double> &feasible_kapp,
                                const std::vector<double> &feasible_lamb,
                                double thickness,
                                double poisson_ratio,
                                double beta);