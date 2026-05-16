/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include "LocalGlobalSolver.h"

#include <TinyAD/Support/GeometryCentral.hh>
#include <TinyAD/ScalarFunction.hh>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include <limits>

Eigen::MatrixXd parameterization(const Eigen::MatrixXd& V,
                                 Eigen::MatrixXi& F,
                                 double lambda1,
                                 double lambda2,
                                 double wD = 0,
                                 int n_iter = 1000,
                                 double lim = 1e-6);

void parameterization(const Eigen::MatrixXd& V,
                      Eigen::MatrixXd& P,
                      const Eigen::MatrixXi& F,
                      double lambda1,
                      double lambda2,
                      double wD,
                      int n_iter,
                      double lim);

geometrycentral::surface::FaceData<Eigen::Matrix2d>
precomputeParamData(geometrycentral::surface::VertexPositionGeometry& geometry);

geometrycentral::surface::FaceData<Eigen::Matrix2d>
precomputeSimData(geometrycentral::surface::ManifoldSurfaceMesh& mesh,
                  const Eigen::MatrixXd& P,
                  const Eigen::MatrixXi& F);

geometrycentral::surface::EdgeData<double>
computeDualCotanWeights(geometrycentral::surface::IntrinsicGeometryInterface& geometry);

void subdivideMesh(geometrycentral::surface::VertexPositionGeometry& geometry,
                   Eigen::MatrixXd& V,
                   Eigen::MatrixXd& P,
                   Eigen::MatrixXi& F,
                   std::vector<Eigen::SparseMatrix<double>>& subdivMat,
                   double threshold);

/**
 * Compute tutte embedding with boundary on circle.
 * Per-vertex 2D coordinates returned as geometrycentral VertexData.
 */
Eigen::MatrixXd tutte_embedding(const Eigen::MatrixXd& _V, const Eigen::MatrixXi& _F);

TinyAD::ScalarFunction<2, double, geometrycentral::surface::VertexRangeF::Etype>
parameterizationFunction(geometrycentral::surface::VertexPositionGeometry& geometry,
                         double wPhi,
                         double lambda1,
                         double lambda2);

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
computeSVDdata(const Eigen::MatrixXd& V, const Eigen::MatrixXd& P, const Eigen::MatrixXi& F);

/**
 * Per-face lambda statistics (area-weighted), used to inspect/decide gauge shift.
 * lambda_f = sqrt(0.5 * trace(a)),  a = (V edge matrix * P edge matrix^-1)^T (...)
 *   - lmin, lmax: per-face min/max
 *   - lmean    : area-weighted geometric/arithmetic mean (arithmetic, as in code)
 */
struct LambdaStats {
    double lmin  =  std::numeric_limits<double>::infinity();
    double lmax  = -std::numeric_limits<double>::infinity();
    double lmean =  0.0;
};
LambdaStats computeLambdaStats(const Eigen::MatrixXd& V,
                               const Eigen::MatrixXi& F,
                               const Eigen::MatrixXd& P);

/**
 * Optimal gauge scale t*:
 *   t* = argmin_t  SUM_f  A_f * (t*lam_f - clip(t*lam_f, lam_min, lam_max))^2
 * The caller is expected to apply  P /= t*  to land the lambda distribution
 * in the material window [lam_min, lam_max].  V is *not* touched.
 * Both V and P should already be in their final (physical) scale so the
 * lambda statistics reflect the as-printed geometry.
 */
double computeGaugeShiftScale(const Eigen::MatrixXd& V,
                              const Eigen::MatrixXi& F,
                              const Eigen::MatrixXd& P,
                              double lambda_min,
                              double lambda_max);