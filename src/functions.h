#pragma once

#include <TinyAD/Support/GeometryCentral.hh>
#include <TinyAD/ScalarFunction.hh>
#include <geometrycentral/surface/intrinsic_geometry_interface.h>




TinyAD::ScalarFunction<3, double, geometrycentral::surface::VertexRangeF::Etype>
simulationFunction(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
                                                             const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
                                                             const geometrycentral::surface::FaceData<double>& lambda,
                                                             const geometrycentral::surface::VertexData<double>& kappa,
                                                             double E,
                                                             double nu,
                                                             double h,
                                                             double w_s,
                                                             double w_b);

TinyAD::ScalarFunction<3, double, geometrycentral::surface::VertexRangeF::Etype>
    simulationFunction(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
                   const geometrycentral::surface::VertexData<bool>& is_boundary_vertex,
                   const geometrycentral::surface::FaceData<bool>& is_boundary_face,
                   const geometrycentral::surface::FaceData<int>& boundary_ref_index,
                       const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
                       const geometrycentral::surface::FaceData<double>& lambda,
                       const geometrycentral::surface::FaceData<double>& kappa,
                       double E,
                       double nu,
                       double h,
                       double w_s,
                       double w_b);

TinyAD::ScalarFunction<3, double, geometrycentral::surface::VertexRangeF::Etype>
simulationFunction(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
                   const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
                   const geometrycentral::surface::VertexData<double>& lambda,
                   const geometrycentral::surface::VertexData<double>& kappa,
                   double E,
                   double nu,
                   double h,
                   double w_s,
                   double w_b);

TinyAD::ScalarFunction<1, double, Eigen::Index>
    adjointFunction(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
                    const Eigen::MatrixXi& F,
                    const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
                    const geometrycentral::surface::FaceData<double>& lambda,
                    double E,
                    double nu,
                    double h,
                    double w_s,
                    double w_b);

                    TinyAD::ScalarFunction<1, double, Eigen::Index>
adjointFunction(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
                const Eigen::MatrixXi& F,
                const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
                const geometrycentral::surface::VertexData<double>& kappa,
                double E,
                double nu,
                double h,
                double w_s,
                double w_b);

TinyAD::ScalarFunction<1, double, Eigen::Index>
adjointFunction(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
                const Eigen::MatrixXi& F,
                const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
                double E,
                double nu,
                double h,
                double w_s,
                double w_b);



