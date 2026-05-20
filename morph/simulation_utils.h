#pragma once

#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>

#include <geometrycentral/surface/intrinsic_geometry_interface.h>
#include <geometrycentral/surface/surface_mesh.h>

/// Build the lumped vertex mass vector used by SGN distance metric
/// (size = 3 * nV, repeated triple per vertex, normalised by total area).
/// geometry must own a mesh; requireFaceAreas / requireVertexIndices are called.
Eigen::VectorXd computeVertexMasses(geometrycentral::surface::IntrinsicGeometryInterface& geometry);

/// Per-face diagonal mass matrix used by the kappa regulariser (OptKap).
/// Entry i = 0.5 / det(MrInv[f_i]).
Eigen::SparseMatrix<double>
computeFaceMassKappa(geometrycentral::surface::SurfaceMesh& mesh,
                     const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv);

/// Per-face diagonal mass matrix used by the lambda regulariser (OptLam).
/// Entry i = faceAreas[f_i].
Eigen::SparseMatrix<double>
computeFaceMassLambda(geometrycentral::surface::IntrinsicGeometryInterface& geometry);

/// Uniform-weight dual-graph Laplacian on the face-face graph (boundary edges
/// excluded).  Shared by both OptKap and OptLam regularisers.
Eigen::SparseMatrix<double>
computeFaceDualLaplacian(geometrycentral::surface::SurfaceMesh& mesh);

Eigen::SparseMatrix<double> projectionMatrix(const std::vector<int>& fixedIdx, int size);

Eigen::SparseMatrix<double> buildHGN(const Eigen::VectorXd& masses,
                                     const Eigen::SparseMatrix<double>& P,
                                     const Eigen::SparseMatrix<double>& M_theta,
                                     const Eigen::SparseMatrix<double>& H);

void updateHGN(Eigen::SparseMatrix<double>& HGN,
               const Eigen::SparseMatrix<double>& P,
               const Eigen::SparseMatrix<double>& H);

const auto nullexpr = [](double) {};

/// MGDA (Multiple Gradient Descent) combination weight for two directions.
///   alpha = argmin_{a in [0,1]}  || a*d_F + (1-a)*d_P ||_2^2
/// closed form:
///   num   = ||d_P||^2 - <d_F, d_P>
///   denom = ||d_F||^2 + ||d_P||^2 - 2*<d_F, d_P>
///   alpha = clamp(num / denom, 0, 1)
/// Special cases: if denom < 1e-30 (d_F == d_P) return 0.5 (degenerate convex combo).
double mgda_alpha(const Eigen::VectorXd& d_F, const Eigen::VectorXd& d_P);

/// Two-objective Armijo line search.  Returns the largest accepted step
/// size s in [shrink^max_iters, 1] satisfying both
///   F(x0 + s*d) <= F(x0) + c*s*<gF, d>
///   P(x0 + s*d) <= P(x0) + c*s*<gP, d>
/// or -1 if no such s in the search range works.
///
/// Uses a fixed shrink (no polynomial interp): with two competing slopes
/// the cubic model is unreliable, so we keep behaviour predictable.
template <class EvalF, class EvalP, class Callback = decltype(nullexpr)>
double lineSearchMulti(const Eigen::VectorXd& x0,
                       const Eigen::VectorXd& d,
                       const double f0,
                       const Eigen::VectorXd& gF,
                       const double p0,
                       const Eigen::VectorXd& gP,
                       const EvalF& evalF,
                       const EvalP& evalP,
                       const Callback& callback = nullexpr,
                       const double shrink = 0.6,
                       const int max_iters = 32)
{
  const double slopeF = d.dot(gF);
  const double slopeP = d.dot(gP);
  const double c = 1e-4;
  Eigen::VectorXd x_trial = x0;
  double s = 1.0;

  for(int i = 0; i < max_iters; ++i)
  {
    x_trial = x0 + s * d;
    callback(s);
    const double f_new = evalF(x_trial);
    const double p_new = evalP(x_trial);

    const bool okF = (f_new <= f0 + c * s * slopeF);
    const bool okP = (p_new <= p0 + c * s * slopeP);
    if(okF && okP)
      return s;

    s *= shrink;
  }
  return -1;
}

template <class Func, class Callback = decltype(nullexpr)>
double lineSearch(const Eigen::VectorXd& x0,
                  const Eigen::VectorXd& d,
                  const double f,
                  const Eigen::VectorXd& g,
                  const Func& eval,
                  const Callback& callback = nullexpr,
                  const double shrink = 0.6,
                  const int max_iters = 16)
{
  const double slope = d.dot(g);  // directional derivative; negative for a descent direction
  Eigen::VectorXd x_trial = x0;
  double s = 1.0;
  double s_prev = 1.0;
  double f_prev = f;

  for(int i = 0; i < max_iters; ++i)
  {
    x_trial = x0 + s * d;
    callback(s);
    const double f_new = eval(x_trial);

    if(f_new <= f + 1e-4 * s * slope)  // Armijo condition
      return s;

    // Choose next step via polynomial interpolation rather than fixed shrink.
    // This finds a good step in ~1-2 probes instead of log(s*)/log(shrink) probes.
    double s_next;
    if(i == 0)
    {
      // Quadratic interpolation through (0,f) with slope, and (1, f_new).
      // p(s) = a*s^2 + slope*s + f  =>  a = f_new - f - slope  (always > 0 when Armijo fails)
      const double a = f_new - f - slope;
      s_next = (a > 1e-15) ? std::clamp(-0.5 * slope / a, 0.1, 0.9) : s * shrink;
    }
    else
    {
      // Cubic interpolation through (0,f) with slope, (s_prev, f_prev), (s, f_new).
      const double denom = s_prev * s_prev * s * s * (s_prev - s);
      if(std::abs(denom) < 1e-30)
      {
        s_next = s * shrink;
      }
      else
      {
        const double A = (s * s * (f_prev - f - slope * s_prev)
                        - s_prev * s_prev * (f_new - f - slope * s)) / denom;
        const double B = (s_prev * s_prev * s_prev * (f_new - f - slope * s)
                        - s * s * s * (f_prev - f - slope * s_prev)) / denom;
        const double disc = B * B - 3.0 * A * slope;
        if(std::abs(A) < 1e-15)
          s_next = (std::abs(B) > 1e-15)
                     ? std::clamp(-slope / (2.0 * B), 0.1 * s, 0.9 * s)
                     : s * shrink;
        else if(disc >= 0.0)
          s_next = std::clamp((-B + std::sqrt(disc)) / (3.0 * A), 0.1 * s, 0.9 * s);
        else
          s_next = s * shrink;
      }
    }

    s_prev = s;
    f_prev = f_new;
    s = s_next;
  }
  return -1;
}

std::vector<int> findCenterFaceIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F);
std::vector<int> findCenterVertexIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F);
    std::vector<int> findCornerFaceIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F);

std::vector<int> findCornerVertexIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F);
    //std::vector<int> findCornerVertexIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F);