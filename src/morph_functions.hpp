#pragma once


#include <TinyAD/Support/GeometryCentral.hh>
#include <TinyAD/ScalarFunction.hh>

#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

geometrycentral::surface::FaceData<Eigen::Matrix2d> precomputeMrInv(geometrycentral::surface::ManifoldSurfaceMesh& mesh,
                                                                    const Eigen::MatrixXd& P,
                                                                    const Eigen::MatrixXi& F);

geometrycentral::surface::FaceData<Eigen::MatrixXd> precomputeM(geometrycentral::surface::ManifoldSurfaceMesh& mesh,
                                                                    const Eigen::MatrixXd& P,
                                                                    const Eigen::MatrixXi& F);