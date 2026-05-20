#pragma once

#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>

Eigen::SparseMatrix<double> projectionMatrix(const std::vector<int>& fixedIdx, int size);

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
                  const Func& eval,
                  const Callback& callback = nullexpr,
                  const double shrink = 0.6,
                  const int max_iters = 32)
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