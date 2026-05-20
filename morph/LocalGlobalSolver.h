#pragma once

#include "solvers.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <memory>

class LocalGlobalSolver
{
public:
  LocalGlobalSolver(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
  void init(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

  void solveOneStep(Eigen::Ref<Eigen::MatrixX2d> U, double sMin, double sMax);

  void solve(Eigen::Ref<Eigen::MatrixX2d> U, double sMin, double sMax, int nbIter = -1);

  /// Per-face overloads: clamp each face's projected SVD to its own
  /// [sMin_pf(i), sMax_pf(i)] window.  Use this for lambda-aware ARAP
  /// where each face has its own target stretch ratio.
  void solveOneStep(Eigen::Ref<Eigen::MatrixX2d> U,
                    const Eigen::VectorXd& sMin_pf,
                    const Eigen::VectorXd& sMax_pf);

  void solve(Eigen::Ref<Eigen::MatrixX2d> U,
             const Eigen::VectorXd& sMin_pf,
             const Eigen::VectorXd& sMax_pf,
             int nbIter = -1);

  Eigen::VectorXd stretchAngles();

  Eigen::VectorXd s1;
  Eigen::VectorXd s2;
  Eigen::MatrixX2d stressX;
  Eigen::MatrixX2d stressY;

  int nV;
  int nF;

protected:
  virtual Eigen::Matrix2d project(const Eigen::Matrix2d& M, double sMin, double sMax, int i);

  void buildRHS(const Eigen::Ref<Eigen::MatrixX2d> V, const Eigen::Ref<Eigen::MatrixX3i> F);

  void buildLinearBlock(const Eigen::Ref<Eigen::MatrixXd> V,
                        const Eigen::Ref<Eigen::MatrixXi> F,
                        const int d,
                        Eigen::SparseMatrix<double>& Kd);

private:
  Eigen::SparseMatrix<double> _K; // rhs pre-multiplier
  Eigen::Matrix<double, 2, -1> _Ainv;
  Eigen::MatrixX3i _F;
  std::unique_ptr<LDLTSolver> _solver;
};
