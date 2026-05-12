#pragma once

#include <TinyAD/Support/GeometryCentral.hh>
#include <TinyAD/ScalarFunction.hh>
#include <geometrycentral/surface/intrinsic_geometry_interface.h>

#include <vector>

// Per-face Young's modulus (E_face) replaces the old scalar `E`.
// Callers are expected to supply E_face already normalized by a reference
// modulus so that the numerical regime of w_s / w_b / epsilon stays stable
// across materials with heterogeneous stiffness.
TinyAD::ScalarFunction<3, double, geometrycentral::surface::VertexRangeF::Etype>
simulationFunction(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                   const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                   const geometrycentral::surface::FaceData<double> &lambda,
                   const geometrycentral::surface::FaceData<double> &kappa,
                   const geometrycentral::surface::FaceData<double> &E_face,
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
                   const geometrycentral::surface::FaceData<double> &E_face,
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
                   const geometrycentral::surface::FaceData<double> &E_face,
                   double nu,
                   double h,
                   double w_s,
                   double w_b,
                   const std::vector<int> &ref_faces);

// Forward declaration for material curve
struct M_Poly_Curve;

TinyAD::ScalarFunction<1, double, Eigen::Index>
adjointFunction_FixLam_OptKap(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                              const Eigen::MatrixXi &F,
                              const geometrycentral::surface::FaceData<Eigen::Matrix2d> &MrInv,
                              const geometrycentral::surface::FaceData<double> &lambda,
                              const geometrycentral::surface::FaceData<double> &E_face,
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
                              const geometrycentral::surface::FaceData<double> &E_face,
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
                              const geometrycentral::surface::FaceData<double> &E_face,
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
                               const geometrycentral::surface::FaceData<double> &E_face,
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

/// Joint (lambda, kappa) feasibility penalty using a Lorentzian soft-min over
/// the discrete feasible set, with strict Saint-Venant elastic energy as the
/// per-pair distance metric (Efrati 2009 non-Euclidean plate theory).
///
/// Per-face SV elastic energy distance to feasible_j:
///     a    = h / 4
///     b    = h^3 / 12
///     d²_E = E_face[f] / (1-nu) * [ a * (lambda^2 - lambda_j^2)^2
///                                 + b * (kappa - kappa_j)^2 ]
///
/// Lorentzian soft-min over all 49 feasible points (no commitment):
///     penalty = -log( sum_j 1 / (1 + d²_E_j * well_scale) ) / nF
///
/// Replaces both the original log-sum-exp `betaP` form (whose well depth
/// vanished at beta=50, leaving the penalty essentially flat) and the
/// transitional hard-min variant (which committed to one feasible per
/// face per stage, suboptimal when the chosen target was not the GT cell).
/// The Lorentzian + SV-energy combination keeps the assignment soft so the
/// continuous solution can drift across Voronoi boundaries while the
/// penalty operates on a physically correct elastic-energy metric.
///
/// `well_scale` controls the well sharpness; default ~30000 for h=1.
TinyAD::ScalarFunction<1, double, Eigen::Index>
JointMaterialPenaltyPerF_OptKap(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                const Eigen::MatrixXi &F,
                                const geometrycentral::surface::FaceData<double> &lambda_pf,
                                const geometrycentral::surface::FaceData<double> &E_face,
                                const std::vector<double> &feasible_lamb,
                                const std::vector<double> &feasible_kapp,
                                double thickness,
                                double poisson_ratio,
                                double well_scale);

TinyAD::ScalarFunction<1, double, Eigen::Index>
JointMaterialPenaltyPerF_OptLam(geometrycentral::surface::IntrinsicGeometryInterface &geometry,
                                const Eigen::MatrixXi &F,
                                const geometrycentral::surface::FaceData<double> &kappa_pf,
                                const geometrycentral::surface::FaceData<double> &E_face,
                                const std::vector<double> &feasible_lamb,
                                const std::vector<double> &feasible_kapp,
                                double thickness,
                                double poisson_ratio,
                                double well_scale);
