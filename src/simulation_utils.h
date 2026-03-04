#pragma once

#include <Eigen/SparseCore>
#include <geometrycentral/surface/intrinsic_geometry_interface.h>
#include <geometrycentral/surface/surface_mesh.h>

Eigen::SparseMatrix<double> projectionMatrix(const std::vector<int>& fixedIdx, int size);

// Build cotan Laplacian from flat reference domain (using MrInv) instead of target surface
Eigen::SparseMatrix<double> buildFlatCotanLaplacian(
    geometrycentral::surface::IntrinsicGeometryInterface& geometry,
    const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv);

Eigen::SparseMatrix<double> buildHGN(const Eigen::VectorXd& masses,
                                     const Eigen::SparseMatrix<double>& P,
                                     const Eigen::SparseMatrix<double>& M_theta,
                                     const Eigen::SparseMatrix<double>& H);

void updateHGN(Eigen::SparseMatrix<double>& HGN,
               const Eigen::SparseMatrix<double>& P,
               const Eigen::SparseMatrix<double>& H);

const auto nullexpr = [](double) {};

template <class Func, class Callback = decltype(nullexpr)>
double lineSearch(const Eigen::VectorXd& x0,
                  const Eigen::VectorXd& d,
                  const double f,
                  const Eigen::VectorXd& g,
                  const Func& eval, // Callable of type T(const Eigen::Vector<T, d>&)
                  const Callback& callback = nullexpr,
                  const double shrink = 0.6,
                  const int max_iters = 32)
{
  Eigen::VectorXd x_new = x0;
  double s = 1.0;
  for(int i = 0; i < max_iters; ++i)
  {
    x_new = x0 + s * d;
    callback(s);
    const double f_new = eval(x_new);
    if(f_new <= f + 1e-4 * s * d.dot(g)) // Armijo condition
      return s;

    s *= shrink;
  }
  // line search couldn't find improvement
  return -1;
}

std::vector<int> findCenterFaceIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F);
std::vector<int> findCenterVertexIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F);
    std::vector<int> findCornerFaceIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F);

std::vector<int> findCornerVertexIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F);
    //std::vector<int> findCornerVertexIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F);