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
// Adjoint function for OptP: variables are [x (3|V|), P (2|V|)], constants
// are (lambda_pf, kappa_pf).  Same non-Euclidean plate energy as the
// forward simulation, but expressed so that P enters through Mr = [P1-P0,
// P2-P0] per face -> MrInv -> F, dA, and TinyAD autodiffs through it.
// Used by sparse_gauss_newton_FixMaterial_OptP to build the SGN KKT
// system for distance-driven P-update (replaces ARAP P-update).
TinyAD::ScalarFunction<1, double, Eigen::Index>
adjointFunction_FixMaterial_OptP(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                 const Eigen::MatrixXi &F,
                                 const geometrycentral::surface::FaceData<double> &lambda_pf,
                                 const geometrycentral::surface::FaceData<double> &kappa_pf,
                                 double E,
                                 double nu,
                                 double h,
                                 double w_s,
                                 double w_b,
                                 const std::vector<int> &ref_faces);


TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialPenaltyFunctionPerV(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                            const std::vector<double> &feasible_vals,
                            double beta);

TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialPenaltyFunctionPerF(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                            const std::vector<double> &feasible_vals,
                            double beta);

// 2D-joint feasibility penalty based on Efrati non-Euclidean plate
// elastic-strain-energy distance.  For each face, compute
//   d^2_j(lambda, kappa) = lambda_bar_j^2 * (E/(1-nu)) *
//       [ (h/4) * (lambda^2 - lambda_bar_j^2)^2
//       + (h^3/12) * (kappa - kappa_bar_j)^2 ]
// where (lambda_bar_j, kappa_bar_j) is the j-th feasible material pair
// (sourced from feasible_lamb[j] / feasible_kapp[j]).  The penalty per
// face is (beta/nF) * min_j d^2_j; total penalty is summed over faces
// and the TinyAD scalar function reports gradient/Hessian w.r.t. the
// *active* variable (kappa when self_is_kappa, lambda otherwise);
// the other variable enters as a per-face constant from other_const.
//
// other_const must have length == nF and is captured by value.
TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialPenaltyFunctionPerF_Efrati(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                   const std::vector<double> &cand_self,
                                   const std::vector<double> &cand_other,
                                   const std::vector<double> &other_const,
                                   double E,
                                   double nu,
                                   double h,
                                   double beta,
                                   bool self_is_kappa);

// 2D-joint feasibility penalty using a plain Euclidean distance² in
// (lambda, kappa) space (no physical weighting):
//   d²_j = (lambda - lambda_bar_j)² + (kappa - kappa_bar_j)²
//   penalty per face = (beta/nF) * min_j d²_j
// The TinyAD scalar function differentiates w.r.t. the active variable
// (kappa when self_is_kappa, lambda otherwise); the other variable
// enters as a per-face constant from other_const.
TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialPenaltyFunctionPerF_Joint2D(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                    const std::vector<double> &cand_self,
                                    const std::vector<double> &cand_other,
                                    const std::vector<double> &other_const,
                                    double beta,
                                    bool self_is_kappa);