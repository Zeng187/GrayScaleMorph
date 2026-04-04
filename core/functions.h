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
                   double w_b,
                   const std::vector<int> &ref_faces);

// Forward declaration for material curve
struct M_Poly_Curve;

// Simulation function with material-based lambda/kappa computation
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
                               double w_b,
                               const std::vector<int> &ref_faces);

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
                               double w_b,
                               const std::vector<int> &ref_faces);

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
                              double w_b,
                              const std::vector<int> &ref_faces);

TinyAD::ScalarFunction<1, double, Eigen::Index>
adjointFunction_FixKap_OptLam(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                              const Eigen::MatrixXi &F,
                              const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                              const geometrycentral::surface::VertexData<double> &kappa,
                              double E,
                              double nu,
                              double h,
                              double w_s,
                              double w_b,
                              const std::vector<int> &ref_faces);

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
                                 double w_b,
                                 const std::vector<int> &ref_faces);

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
                                 double w_b,
                                 const std::vector<int> &ref_faces);

// Material penalty functions (unchanged — no L computation)
TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialPenaltyFunctionPerV(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                            const std::vector<double> &feasible_vals,
                            double beta);

TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialPenaltyFunctionPerF(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                            const std::vector<double> &feasible_vals,
                            double beta);

TinyAD::ScalarFunction<1, double, Eigen::Index>
JointMaterialPenaltyPerF_OptKap(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                const Eigen::MatrixXi &F,
                                const geometrycentral::surface::FaceData<double> &lambda_pf,
                                const std::vector<double> &feasible_lamb,
                                const std::vector<double> &feasible_kapp,
                                double beta);

TinyAD::ScalarFunction<1, double, Eigen::Index>
JointMaterialPenaltyPerF_OptLam(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                const Eigen::MatrixXi &F,
                                const geometrycentral::surface::FaceData<double> &kappa_pf,
                                const std::vector<double> &feasible_lamb,
                                const std::vector<double> &feasible_kapp,
                                double beta);
