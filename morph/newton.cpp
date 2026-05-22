#include "newton.h"
#include "functions.h"
#include "morph_functions.hpp"
#include "parameterization.h"
#include "simulation_utils.h"
#include "solvers.h"
#include "timer.h"

#include <TinyAD/Utils/NewtonDecrement.hh>
#include <cmath>
#include <limits>

using namespace geometrycentral::surface;

template <class Func, class Solver>
void newton(Eigen::VectorXd& x,
            Func& func,
            Solver& solver,
            int max_iters,
            double lim,
            bool verbose,
            const std::vector<int>& fixedIdx,
            const std::function<void(const Eigen::VectorXd&)>& callback)
{
  Timer timer("Newton", !verbose);

  if(verbose)
    std::cout << "Initial newton energy: " << func.eval(x) << std::endl;

  Eigen::SparseMatrix<double> P = projectionMatrix(fixedIdx, x.size());

  for(int i = 0; i < max_iters; ++i)
  {
    auto [f, g, H] = func.eval_with_derivatives(x);

    for(int j = 0; j < H.cols(); ++j)
      H.coeffRef(j, j) += 1e-10;

    // restrict H and g to free variables
    H = (P * H * P.transpose()).eval();
    g = P * g;

    // Newton direction
    if(i == 0)
      solver.compute(H);
    else
      solver.factorize(H);

    bool exact = true;
    if(solver.info() != Eigen::Success)
    {
      exact = false;
      auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
      H_proj = (P * H_proj * P.transpose()).eval();

      H = 0.9 * H + 0.1 * H_proj;
      solver.factorize(H);
      if(solver.info() != Eigen::Success)
        solver.factorize(H_proj);
    }

    Eigen::VectorXd d = -solver.solve(g);

    if(verbose)
    {
      if(exact)
        std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(d, g)
                  << "\tFactorization = Exact\n";
      else
        std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(d, g)
                  << "\tFactorization = Project\n";
    }

    d = P.transpose() * d;
    g = P.transpose() * g;

    double s = lineSearch(x, d, f, g, func);
    if(s < 0)
      break;
    x += s * d;

    if(TinyAD::newton_decrement(d, g) < lim && exact)
      break;

    callback(x);
  }
  if(verbose)
    std::cout << "Final newton energy: " << func.eval(x) << "\n";
}

template <class Func>
void newton(IntrinsicGeometryInterface& geometry,
            Eigen::MatrixXd& V,
            Func& func,
            int max_iters,
            double lim,
            bool verbose,
            const std::vector<int>& fixedIdx,
            const std::function<void(const Eigen::VectorXd&)>& callback)
{
  // Assemble inital x vector
  geometry.requireVertexIndices();
  Eigen::VectorXd x = func.x_from_data([&](Vertex v) { return V.row(geometry.vertexIndices[v]); });

  LLTSolver solver;

  // Newton algorithm
  newton(x, func, solver, max_iters, lim, verbose, fixedIdx, callback);

  func.x_to_data(x, [&](Vertex v, const auto& row) { V.row(geometry.vertexIndices[v]) = row; });
}


Eigen::MatrixXd sparse_gauss_newton_FixLam_OptKap(IntrinsicGeometryInterface& geometry,
                                    const Eigen::MatrixXd& targetV,
                                    const Eigen::MatrixXd& initV,
                                    const FaceData<Eigen::Matrix2d>& MrInv,
                                    FaceData<double>& theta1,
                                    FaceData<double>& theta2,
                                    const Eigen::VectorXd& masses,
                                    double other_reg,
                                    const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
                                    const std::vector<int>& fixedIdx,
                                    int max_iters,
                                    double lim,
                                    double wM,
                                    double wL,
                                    double E,
                                    double nu,
                                    double h,
                                    double w_s,
                                    double w_b,
                                    const std::vector<int>& ref_faces,
                                    double& final_distance,
                                    double& final_spn_energy,
                                    double& final_self_reg,
                                    const std::function<void(const Eigen::VectorXd&)>& callback,
                                    const SgnIterCallback& iter_cb)
{
  geometry.requireFaceAreas();
  geometry.requireVertexIndices();

  SurfaceMesh& mesh = geometry.mesh;

  // Per-face regularization for theta2 (kappa, |F|)
  const size_t nF = mesh.nFaces();

  // Diagonal mass matrix on faces (flat-area-weighted)
  Eigen::SparseMatrix<double> M_theta(nF, nF);
  M_theta.reserve(nF);
  {
    size_t iF = 0;
    for(Face f : mesh.faces())
    {
      M_theta.insert(iF, iF) = 0.5 / MrInv[f].determinant();
      ++iF;
    }
  }

  // Face-dual graph Laplacian (uniform weights)
  Eigen::SparseMatrix<double> L(nF, nF);
  {
    FaceData<size_t> faceIdx(mesh);
    size_t cnt = 0;
    for(Face f : mesh.faces()) faceIdx[f] = cnt++;
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(mesh.nEdges() * 4);
    std::vector<double> diag(nF, 0.0);
    for(Edge e : mesh.edges())
    {
      if(e.isBoundary()) continue;
      Halfedge he = e.halfedge();
      size_t i = faceIdx[he.face()];
      size_t j = faceIdx[he.twin().face()];
      trips.emplace_back((int)i, (int)j, -1.0);
      trips.emplace_back((int)j, (int)i, -1.0);
      diag[i] += 1.0; diag[j] += 1.0;
    }
    for(size_t i = 0; i < nF; ++i)
      trips.emplace_back((int)i, (int)i, diag[i]);
    L.setFromTriplets(trips.begin(), trips.end());
  }

  Eigen::VectorXd theta = theta2.toVector();
  Eigen::VectorXd xTarget(targetV.size());
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      xTarget(3 * i + j) = targetV(i, j);
  Eigen::VectorXd x(initV.size());
  for(int i = 0; i < initV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      x(3 * i + j) = initV(i, j);

  LLTSolver adjointSolver;


  auto distance = [&](const Eigen::VectorXd& th) {
    theta2.fromVector(th);
    auto simFunc = simulationFunction(geometry, MrInv, theta1, theta2, E,nu,h,w_s,w_b, ref_faces);
    newton(x, simFunc, adjointSolver, 100, lim, false, fixedIdx);

    // Unified SPN energy = distance + self regulariser + other-variable regulariser (constant in this stage)
    return (x - xTarget).dot(masses.cwiseProduct(x - xTarget)) + wM * th.dot(M_theta * th) + wL * th.dot(L * th) + other_reg;
  };

  // Build matrix P
  Eigen::SparseMatrix<double> P = projectionMatrix(fixedIdx, x.size());
  // Hessian matrix H
  Eigen::VectorXd X(targetV.size() + theta.size());
  X.head(targetV.size()) = x;
  X.tail(theta.size()) = theta;
  Eigen::SparseMatrix<double> H = adjointFunc.eval_hessian(X);

  // Build HGN matrix
  Eigen::SparseMatrix<double> HGN = buildHGN(2 * masses, P, 2 * wM * M_theta + 2 * wL * L, H);

  auto distanceGrad = [&](const Eigen::VectorXd& th) -> Eigen::VectorXd {
    Eigen::VectorXd X(targetV.size() + th.size());
    X.head(targetV.size()) = x;
    X.tail(th.size()) = th;
    H = adjointFunc.eval_hessian(X);

    for(int j = 0; j < targetV.size(); ++j)
      H.coeffRef(j, j) += 1e-10;

    Eigen::SparseMatrix<double> A = (P * H.block(0, 0, targetV.size(), targetV.size()) * P.transpose()).eval();

    adjointSolver.factorize(A);
    if(adjointSolver.info() != Eigen::Success)
    {
      auto [f, g, A_proj] = adjointFunc.eval_with_hessian_proj(X);
      A_proj = (P * A_proj.block(0, 0, targetV.size(), targetV.size()) * P.transpose()).eval();

      A = 0.9 * A + 0.1 * A_proj;
      adjointSolver.factorize(A);
      if(adjointSolver.info() != Eigen::Success)
        adjointSolver.factorize(A_proj);
    }

    Eigen::VectorXd b = P * masses.cwiseProduct(x - xTarget);
    Eigen::VectorXd dir = adjointSolver.solve(b);
    if(adjointSolver.info() != Eigen::Success)
      std::cout << "Solver error\n";

    dir = P.transpose() * dir;

    return -2 * H.block(targetV.size(), 0, th.size(), targetV.size()) * dir + 2 * wM * M_theta * th + 2 * wL * L * th;
  };



  double energy = distance(theta);
  std::cout << "Initial SPN energy: " << energy << " " << energy<<"\t distance: "<<(x - xTarget).dot(masses.cwiseProduct(x - xTarget))
            << std::endl;

  LUSolver solver;

  for(int i = 0; i < max_iters; ++i)
  {
    double f = distance(theta);
    Eigen::VectorXd g = distanceGrad(theta);

    Eigen::VectorXd b(2 * x.size() - 2 * fixedIdx.size() + theta.size());
    b.setZero();
    b.segment(x.size() - fixedIdx.size(), theta.size()) = -g;

    // Update HGN
    updateHGN(HGN, P, H);

    if(i == 0)
      solver.compute(HGN);
    else
      solver.factorize(HGN);

    if(solver.info() != Eigen::Success)
    {
      std::cout << "Solver error\n";
      return targetV;
    }

    Eigen::VectorXd d = solver.solve(b);
    Eigen::VectorXd deltaTheta = d.segment(x.size() - fixedIdx.size(), theta.size());
    Eigen::VectorXd deltaX = d.segment(0, x.size() - fixedIdx.size());
    deltaX = P.transpose() * deltaX;



    // LINE SEARCH
    Eigen::VectorXd x_old = x;
    double s = lineSearch(theta, deltaTheta, f, g, distance, [&](double s) { x = x_old + s * deltaX; });
    if(s < 0)
    {
      // lineSearch's 32 shrinks each call eval() which runs forward newton
      // and can drag x into a different equilibrium basin.  When the search
      // ultimately fails, x is left in a contaminated state (not at
      // equilibrium for the current theta).  Revert x to the last-good
      // equilibrium (x_old) so the subsequent Final SPN distance() call
      // converges in the correct basin instead of jumping out to a wrong
      // local minimum.
      x = x_old;
      std::cout << "Line search failed (x reverted to last good state)\n";
      break;
    }
    theta += s * deltaTheta;

    const double iter_F   = distance(theta);
    const double iter_d   = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    const double iter_sr  = wM * theta.dot(M_theta * theta) + wL * theta.dot(L * theta);
    std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaTheta, g)
              << "\tSPN energy: " << iter_F
              << "\tDistance: " << iter_d
              << "\tStep size: " << s << std::endl;
    iter_cb(i, x, iter_F, iter_d, iter_sr, /*penalty=*/0.0);
    if(TinyAD::newton_decrement(deltaTheta, g) < lim || solver.info() != Eigen::Success)
      break;

    callback(x);
  }

  // Force final forward-sim convergence — see other variants for rationale.
  const double final_energy = distance(theta);
  final_distance   = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
  final_spn_energy = final_energy;
  final_self_reg   = wM * theta.dot(M_theta * theta) + wL * theta.dot(L * theta);

  Eigen::MatrixXd V(targetV.rows(), 3);
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      V(i, j) = x(3 * i + j);

  theta2.fromVector(theta);
  return V;
}



Eigen::MatrixXd sparse_gauss_newton_FixKap_OptLam(IntrinsicGeometryInterface& geometry,
                                  const Eigen::MatrixXd& targetV,
                                  const Eigen::MatrixXd& initV,
                                  const FaceData<Eigen::Matrix2d>& MrInv,
                                  FaceData<double>& theta1,
                                  FaceData<double>& theta2,
                                  const Eigen::VectorXd& masses,
                                  double other_reg,
                                  const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
                                  const std::vector<int>& fixedIdx,
                                  int max_iters,
                                  double lim,
                                  double wM,
                                  double wL,
                                  double E,
                                  double nu,
                                  double h,
                                  double w_s,
                                  double w_b,
                                  const std::vector<int>& ref_faces,
                                  double& final_distance,
                                  double& final_spn_energy,
                                  double& final_self_reg,
                                  const std::function<void(const Eigen::VectorXd&)>& callback,
                                  const SgnIterCallback& iter_cb)
{
  SurfaceMesh& mesh = geometry.mesh;

  // ----------------------------
  // Regularization for per-face theta1 (lambda)
  // ----------------------------
  const size_t nF = mesh.nFaces();

  // Diagonal mass matrix on faces (area-weighted)
  Eigen::SparseMatrix<double> M_theta(nF, nF);
  M_theta.reserve(nF);
  {
    size_t iF = 0;
    for(Face f : mesh.faces())
    {
      M_theta.insert(iF, iF) = geometry.faceAreas[f];
      ++iF;
    }
  }

  // Simple face graph Laplacian (uniform weights on dual graph)
  // NOTE: If you have better weights (shared-edge length, dihedral, etc.), replace w=1.0.
  Eigen::SparseMatrix<double> L(nF, nF);
  {
    FaceData<size_t> faceIdx(mesh);
    size_t cnt = 0;
    for(Face f : mesh.faces()) faceIdx[f] = cnt++;

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(mesh.nEdges() * 4);
    std::vector<double> diag(nF, 0.0);

    for(Edge e : mesh.edges())
    {
      if(e.isBoundary()) continue;

      Halfedge he = e.halfedge();
      Face f0 = he.face();
      Face f1 = he.twin().face();

      size_t i = faceIdx[f0];
      size_t j = faceIdx[f1];

      const double w = 1.0;
      trips.emplace_back((int)i, (int)j, -w);
      trips.emplace_back((int)j, (int)i, -w);
      diag[i] += w;
      diag[j] += w;
    }

    for(size_t i = 0; i < nF; ++i)
      trips.emplace_back((int)i, (int)i, diag[i]);

    L.setFromTriplets(trips.begin(), trips.end());
  }

  // theta is now size |F|
  Eigen::VectorXd theta = theta1.toVector();

  // pack target and init x (still size 3|V|)
  Eigen::VectorXd xTarget(targetV.size());
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      xTarget(3 * i + j) = targetV(i, j);

  Eigen::VectorXd x(initV.size());
  for(int i = 0; i < initV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      x(3 * i + j) = initV(i, j);

  LLTSolver adjointSolver;

  auto distance = [&](const Eigen::VectorXd& th) {
    theta1.fromVector(th);

    // IMPORTANT: call the overload with (FaceData<double> lambda, VertexData<double> kappa)
    auto simFunc = simulationFunction(geometry, MrInv, theta1, theta2, E, nu, h, w_s, w_b, ref_faces);

    newton(x, simFunc, adjointSolver, 100, lim, false, fixedIdx);

    // Unified SPN energy = distance + self regulariser + other-variable regulariser (constant in this stage)
    return (x - xTarget).dot(masses.cwiseProduct(x - xTarget))
           + wM * th.dot(M_theta * th)
           + wL * th.dot(L * th)
           + other_reg;
  };

  // Build matrix P (still for fixed vertex positions in x)
  Eigen::SparseMatrix<double> P = projectionMatrix(fixedIdx, x.size());

  // Hessian matrix H from adjoint function
  Eigen::VectorXd X(targetV.size() + theta.size());
  X.head(targetV.size()) = x;
  X.tail(theta.size()) = theta;

  Eigen::SparseMatrix<double> H = adjointFunc.eval_hessian(X);

  // Build HGN matrix (replace vertex L by face L)
  Eigen::SparseMatrix<double> HGN = buildHGN(2 * masses, P, 2 * wM * M_theta + 2 * wL * L, H);

  auto distanceGrad = [&](const Eigen::VectorXd& th) -> Eigen::VectorXd {
    Eigen::VectorXd X(targetV.size() + th.size());
    X.head(targetV.size()) = x;
    X.tail(th.size()) = th;

    H = adjointFunc.eval_hessian(X);

    for(int j = 0; j < targetV.size(); ++j)
      H.coeffRef(j, j) += 1e-10;

    Eigen::SparseMatrix<double> A =
        (P * H.block(0, 0, targetV.size(), targetV.size()) * P.transpose()).eval();

    adjointSolver.factorize(A);
    if(adjointSolver.info() != Eigen::Success)
    {
      auto [f, g, A_proj] = adjointFunc.eval_with_hessian_proj(X);
      A_proj =
          (P * A_proj.block(0, 0, targetV.size(), targetV.size()) * P.transpose()).eval();

      A = 0.9 * A + 0.1 * A_proj;
      adjointSolver.factorize(A);
      if(adjointSolver.info() != Eigen::Success)
        adjointSolver.factorize(A_proj);
    }

    Eigen::VectorXd b = P * masses.cwiseProduct(x - xTarget);
    Eigen::VectorXd dir = adjointSolver.solve(b);
    if(adjointSolver.info() != Eigen::Success)
      std::cout << "Solver error\n";

    dir = P.transpose() * dir;

    // gradient wrt theta (size |F|) + regularization
    return -2 * H.block(targetV.size(), 0, th.size(), targetV.size()) * dir
           + 2 * wM * M_theta * th
           + 2 * wL * L * th;
  };

  double energy = distance(theta);
  std::cout << "Initial SPN energy: " << energy << " " << energy <<"\t distance: "<<(x - xTarget).dot(masses.cwiseProduct(x - xTarget))<< std::endl;

  LUSolver solver;

  for(int i = 0; i < max_iters; ++i)
  {
    double f = distance(theta);
    Eigen::VectorXd g = distanceGrad(theta);

    Eigen::VectorXd b(2 * x.size() - 2 * fixedIdx.size() + theta.size());
    b.setZero();
    b.segment(x.size() - fixedIdx.size(), theta.size()) = -g;

    // Update HGN
    updateHGN(HGN, P, H);

    if(i == 0)
      solver.compute(HGN);
    else
      solver.factorize(HGN);

    if(solver.info() != Eigen::Success)
    {
      std::cout << "Solver error\n";
      return targetV;
    }

    Eigen::VectorXd d = solver.solve(b);

    Eigen::VectorXd deltaTheta = d.segment(x.size() - fixedIdx.size(), theta.size());
    Eigen::VectorXd deltaX = d.segment(0, x.size() - fixedIdx.size());
    deltaX = P.transpose() * deltaX;

    // LINE SEARCH
    Eigen::VectorXd x_old = x;
    double s = lineSearch(theta, deltaTheta, f, g, distance,
                          [&](double s) { x = x_old + s * deltaX; });

    if(s < 0)
    {
      // lineSearch's 32 shrinks each call eval() which runs forward newton
      // and can drag x into a different equilibrium basin.  When the search
      // ultimately fails, x is left in a contaminated state (not at
      // equilibrium for the current theta).  Revert x to the last-good
      // equilibrium (x_old) so the subsequent Final SPN distance() call
      // converges in the correct basin instead of jumping out to a wrong
      // local minimum.
      x = x_old;
      std::cout << "Line search failed (x reverted to last good state)\n";
      break;
    }

    theta += s * deltaTheta;

    const double iter_F   = distance(theta);
    const double iter_d   = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    const double iter_sr  = wM * theta.dot(M_theta * theta) + wL * theta.dot(L * theta);
    std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaTheta, g)
              << "\tSPN energy: " << iter_F
              << "\tDistance: " << iter_d
              << "\tStep size: " << s << std::endl;
    iter_cb(i, x, iter_F, iter_d, iter_sr, /*penalty=*/0.0);

    if(TinyAD::newton_decrement(deltaTheta, g) < lim || solver.info() != Eigen::Success)
      break;

    callback(x);
  }

  // Force one final forward-sim convergence on the reverted x so the
  // returned Vr reflects the *actual* stable elastic equilibrium for the
  // current (theta1, theta2) — not the half-converged state of iter N.
  // This makes OptLam-exit distance == OptKap-entry distance (no jump
  // between stages).  May reveal that the last iter's printed "Distance"
  // was misleading (forward sim wasn't fully converged), but that's
  // physically honest.
  const double final_energy = distance(theta);
  final_distance   = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
  final_spn_energy = final_energy;
  final_self_reg   = wM * theta.dot(M_theta * theta) + wL * theta.dot(L * theta);

  Eigen::MatrixXd V(targetV.rows(), 3);
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      V(i, j) = x(3 * i + j);

  theta1.fromVector(theta);
  return V;
}


Eigen::MatrixXd sparse_gauss_newton_FixLam_OptKap_Penalty(IntrinsicGeometryInterface& geometry,
                                    const Eigen::MatrixXd& targetV,
                                    const Eigen::MatrixXd& initV,
                                    const FaceData<Eigen::Matrix2d>& MrInv,
                                    FaceData<double>& theta1,
                                    FaceData<double>& theta2,
                                    const Eigen::VectorXd& masses,
                                    double other_reg,
                                    const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
                                    const TinyAD::ScalarFunction<1, double, Eigen::Index>& penaltyFunc,
                                    const std::vector<int>& fixedIdx,
                                    int max_iters,
                                    double lim,
                                    double wM,
                                    double wL,
                                    double wP,
                                    double E,
                                    double nu,
                                    double h,
                                    double w_s,
                                    double w_b,
                                    const std::vector<int>& ref_faces,
                                    double& final_distance,
                                    double& final_spn_energy,
                                    double& final_self_reg,
                                    const std::function<void(const Eigen::VectorXd&)>& callback)
{
  geometry.requireFaceAreas();
  geometry.requireVertexIndices();

  SurfaceMesh& mesh = geometry.mesh;

  // Per-face regularization for theta2 (kappa, |F|)
  const size_t nF = mesh.nFaces();

  Eigen::SparseMatrix<double> M_theta(nF, nF);
  M_theta.reserve(nF);
  {
    size_t iF = 0;
    for(Face f : mesh.faces())
    {
      M_theta.insert(iF, iF) = 0.5 / MrInv[f].determinant();
      ++iF;
    }
  }

  Eigen::SparseMatrix<double> L(nF, nF);
  {
    FaceData<size_t> faceIdx(mesh);
    size_t cnt = 0;
    for(Face f : mesh.faces()) faceIdx[f] = cnt++;
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(mesh.nEdges() * 4);
    std::vector<double> diag(nF, 0.0);
    for(Edge e : mesh.edges())
    {
      if(e.isBoundary()) continue;
      Halfedge he = e.halfedge();
      size_t i = faceIdx[he.face()];
      size_t j = faceIdx[he.twin().face()];
      trips.emplace_back((int)i, (int)j, -1.0);
      trips.emplace_back((int)j, (int)i, -1.0);
      diag[i] += 1.0; diag[j] += 1.0;
    }
    for(size_t i = 0; i < nF; ++i)
      trips.emplace_back((int)i, (int)i, diag[i]);
    L.setFromTriplets(trips.begin(), trips.end());
  }

  Eigen::VectorXd theta = theta2.toVector();
  Eigen::VectorXd xTarget(targetV.size());
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      xTarget(3 * i + j) = targetV(i, j);
  Eigen::VectorXd x(initV.size());
  for(int i = 0; i < initV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      x(3 * i + j) = initV(i, j);

  LLTSolver adjointSolver;


  auto distance = [&](const Eigen::VectorXd& th) {
    theta2.fromVector(th);
    auto simFunc = simulationFunction(geometry, MrInv, theta1, theta2, E,nu,h,w_s,w_b, ref_faces);
    newton(x, simFunc, adjointSolver, 100, lim, false, fixedIdx);

    double qp = penaltyFunc.eval(th);
    // Unified SPN energy = distance + self regulariser + wP·penalty + other-variable regulariser (constant)
    return (x - xTarget).dot(masses.cwiseProduct(x - xTarget)) + wM * th.dot(M_theta * th) + wL * th.dot(L * th) +
           wP * qp + other_reg;

  };

  // Build matrix P
  Eigen::SparseMatrix<double> P = projectionMatrix(fixedIdx, x.size());
  // Hessian matrix H
  Eigen::VectorXd X(targetV.size() + theta.size());
  X.head(targetV.size()) = x;
  X.tail(theta.size()) = theta;
  Eigen::SparseMatrix<double> H = adjointFunc.eval_hessian(X);
  Eigen::SparseMatrix<double> qH = penaltyFunc.eval_hessian(theta);
  // Build HGN matrix
  Eigen::SparseMatrix<double> HGN = buildHGN(2 * masses, P, 2 * wM * M_theta + 2 * wL * L, H);

  auto distanceGrad = [&](const Eigen::VectorXd& th) -> Eigen::VectorXd {
    Eigen::VectorXd X(targetV.size() + th.size());
    X.head(targetV.size()) = x;
    X.tail(th.size()) = th;
    H = adjointFunc.eval_hessian(X);

    for(int j = 0; j < targetV.size(); ++j)
      H.coeffRef(j, j) += 1e-10;

    Eigen::SparseMatrix<double> A = (P * H.block(0, 0, targetV.size(), targetV.size()) * P.transpose()).eval();

    adjointSolver.factorize(A);
    if(adjointSolver.info() != Eigen::Success)
    {
      auto [f, g, A_proj] = adjointFunc.eval_with_hessian_proj(X);
      A_proj = (P * A_proj.block(0, 0, targetV.size(), targetV.size()) * P.transpose()).eval();

      A = 0.9 * A + 0.1 * A_proj;
      adjointSolver.factorize(A);
      if(adjointSolver.info() != Eigen::Success)
        adjointSolver.factorize(A_proj);
    }

    Eigen::VectorXd b = P * masses.cwiseProduct(x - xTarget);
    Eigen::VectorXd dir = adjointSolver.solve(b);
    if(adjointSolver.info() != Eigen::Success)
      std::cout << "Solver error\n";

    dir = P.transpose() * dir;

    auto [qf, qg] = penaltyFunc.eval_with_gradient(th);
    return -2 * H.block(targetV.size(), 0, th.size(), targetV.size()) * dir + 2 * wM * M_theta * th + 2 * wL * L * th + wP * qg;
  };



  double energy = distance(theta);
  std::cout << "Initial SPN energy: " << energy << " " << energy <<"\t distance: "<<(x - xTarget).dot(masses.cwiseProduct(x - xTarget))
            << std::endl;

  LUSolver solver;

  for(int i = 0; i < max_iters; ++i)
  {
    double f = distance(theta);
    Eigen::VectorXd g = distanceGrad(theta);

    Eigen::VectorXd b(2 * x.size() - 2 * fixedIdx.size() + theta.size());
    b.setZero();
    b.segment(x.size() - fixedIdx.size(), theta.size()) = -g;


    qH = penaltyFunc.eval_hessian(theta);
    HGN = buildHGN(2 * masses, P, 2 * wM * M_theta + 2 * wL * L + wP * qH, H);

    if(i == 0)
      solver.compute(HGN);
    else
      solver.factorize(HGN);

    if(solver.info() != Eigen::Success)
    {
      std::cout << "Solver error\n";
      return targetV;
    }

    Eigen::VectorXd d = solver.solve(b);
    Eigen::VectorXd deltaTheta = d.segment(x.size() - fixedIdx.size(), theta.size());
    Eigen::VectorXd deltaX = d.segment(0, x.size() - fixedIdx.size());
    deltaX = P.transpose() * deltaX;



    // LINE SEARCH
    Eigen::VectorXd x_old = x;
    double s = lineSearch(theta, deltaTheta, f, g, distance, [&](double s) { x = x_old + s * deltaX; });
    if(s < 0)
    {
      // lineSearch's 32 shrinks each call eval() which runs forward newton
      // and can drag x into a different equilibrium basin.  When the search
      // ultimately fails, x is left in a contaminated state (not at
      // equilibrium for the current theta).  Revert x to the last-good
      // equilibrium (x_old) so the subsequent Final SPN distance() call
      // converges in the correct basin instead of jumping out to a wrong
      // local minimum.
      x = x_old;
      std::cout << "Line search failed (x reverted to last good state)\n";
      break;
    }
    theta += s * deltaTheta;

    std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaTheta, g)
              << "\tSPN energy: " << distance(theta)
              << "\tDistance: " << (x - xTarget).dot(masses.cwiseProduct(x - xTarget))
              << "\tStep size: " << s << std::endl;
    if(TinyAD::newton_decrement(deltaTheta, g) < lim || solver.info() != Eigen::Success)
      break;

    callback(x);
  }

  // Force final forward-sim convergence — see other variants for rationale.
  const double final_energy = distance(theta);
  final_distance   = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
  final_spn_energy = final_energy;
  final_self_reg   = wM * theta.dot(M_theta * theta) + wL * theta.dot(L * theta);

  Eigen::MatrixXd V(targetV.rows(), 3);
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      V(i, j) = x(3 * i + j);

  theta2.fromVector(theta);
  return V;
}



Eigen::MatrixXd sparse_gauss_newton_FixKap_OptLam_Penalty(IntrinsicGeometryInterface& geometry,
                                                          const Eigen::MatrixXd& targetV,
                                                          const Eigen::MatrixXd& initV,
                                                          const FaceData<Eigen::Matrix2d>& MrInv,
                                                          FaceData<double>& theta1,   // <-- FaceData lam
                                                          FaceData<double>& theta2,   // FixKap: per-face kappa constant
                                                          const Eigen::VectorXd& masses,
                                                          double other_reg,
                                                          const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
                                                          const TinyAD::ScalarFunction<1, double, Eigen::Index>& penaltyFunc,
                                                          const std::vector<int>& fixedIdx,
                                                          int max_iters,
                                                          double lim,
                                                          double wM,
                                                          double wL,
                                                          double wP,
                                                          double E,
                                                          double nu,
                                                          double h,
                                                          double w_s,
                                                          double w_b,
                                                          const std::vector<int>& ref_faces,
                                                          double& final_distance,
                                                          double& final_spn_energy,
                                                          double& final_self_reg,
                                                          const std::function<void(const Eigen::VectorXd&)>& callback)
{
  geometry.requireFaceAreas();
  geometry.requireVertexIndices();

  SurfaceMesh& mesh = geometry.mesh;

  // ----------------------------
  // face-space regularization for theta1 (|F|)
  // ----------------------------
  const int nF = static_cast<int>(mesh.nFaces());

  // Face area mass matrix
  Eigen::SparseMatrix<double> M_theta(nF, nF);
  M_theta.reserve(nF);
  {
    int iF = 0;
    for(Face f : mesh.faces())
    {
      M_theta.insert(iF, iF) = geometry.faceAreas[f];
      ++iF;
    }
  }

  // Uniform dual-graph Laplacian on faces (replace with your weighted version if desired)
  Eigen::SparseMatrix<double> L(nF, nF);
  {
    FaceData<int> faceIdx(mesh);
    int cnt = 0;
    for(Face f : mesh.faces()) faceIdx[f] = cnt++;

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(mesh.nEdges() * 4);

    std::vector<double> diag(nF, 0.0);

    for(Edge e : mesh.edges())
    {
      if(e.isBoundary()) continue;
      Halfedge he = e.halfedge();
      Face f0 = he.face();
      Face f1 = he.twin().face();

      int i = faceIdx[f0];
      int j = faceIdx[f1];

      const double w = 1.0; // TODO: swap in better weights if needed

      trips.emplace_back(i, j, -w);
      trips.emplace_back(j, i, -w);
      diag[i] += w;
      diag[j] += w;
    }

    for(int i = 0; i < nF; ++i)
      trips.emplace_back(i, i, diag[i]);

    L.setFromTriplets(trips.begin(), trips.end());
  }

  // ----------------------------
  // pack theta/x
  // ----------------------------
  Eigen::VectorXd theta = theta1.toVector();

  Eigen::VectorXd xTarget(targetV.size());
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      xTarget(3 * i + j) = targetV(i, j);

  Eigen::VectorXd x(initV.size());
  for(int i = 0; i < initV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      x(3 * i + j) = initV(i, j);

  LLTSolver adjointSolver;

  // ----------------------------
  // objective
  // ----------------------------
  auto distance = [&](const Eigen::VectorXd& th) {
    theta1.fromVector(th);

    // IMPORTANT: overload with FaceData<double> lambda
    auto simFunc = simulationFunction(geometry, MrInv, theta1, theta2, E, nu, h, w_s, w_b, ref_faces);
    newton(x, simFunc, adjointSolver, 100, lim, false, fixedIdx);

    double qp = penaltyFunc.eval(th);

    // Unified SPN energy = distance + self regulariser + wP·penalty + other-variable regulariser (constant)
    return (x - xTarget).dot(masses.cwiseProduct(x - xTarget))
           + wM * th.dot(M_theta * th)
           + wL * th.dot(L * th)
           + wP * qp
           + other_reg;
  };

  // same as your existing penalty version: x fixed by projection, theta unconstrained
  Eigen::SparseMatrix<double> P = projectionMatrix(fixedIdx, x.size());

  // ----------------------------
  // HGN build
  // ----------------------------
  Eigen::VectorXd X(targetV.size() + theta.size());
  X.head(targetV.size()) = x;
  X.tail(theta.size())   = theta;

  Eigen::SparseMatrix<double> H  = adjointFunc.eval_hessian(X);
  Eigen::SparseMatrix<double> qH = penaltyFunc.eval_hessian(theta);

  Eigen::SparseMatrix<double> HGN =
      buildHGN(2 * masses, P, 2 * wM * M_theta + 2 * wL * L, H);

  auto distanceGrad = [&](const Eigen::VectorXd& th) -> Eigen::VectorXd {
    Eigen::VectorXd X(targetV.size() + th.size());
    X.head(targetV.size()) = x;
    X.tail(th.size())      = th;

    H = adjointFunc.eval_hessian(X);

    for(int j = 0; j < targetV.size(); ++j)
      H.coeffRef(j, j) += 1e-10;

    Eigen::SparseMatrix<double> A =
        (P * H.block(0, 0, targetV.size(), targetV.size()) * P.transpose()).eval();

    adjointSolver.factorize(A);
    if(adjointSolver.info() != Eigen::Success)
    {
      auto [f, g, A_proj] = adjointFunc.eval_with_hessian_proj(X);
      A_proj =
          (P * A_proj.block(0, 0, targetV.size(), targetV.size()) * P.transpose()).eval();

      A = 0.9 * A + 0.1 * A_proj;
      adjointSolver.factorize(A);
      if(adjointSolver.info() != Eigen::Success)
        adjointSolver.factorize(A_proj);
    }

    Eigen::VectorXd b   = P * masses.cwiseProduct(x - xTarget);
    Eigen::VectorXd dir = adjointSolver.solve(b);
    if(adjointSolver.info() != Eigen::Success)
      std::cout << "Solver error\n";

    dir = P.transpose() * dir;

    auto [qf, qg] = penaltyFunc.eval_with_gradient(th);

    return -2 * H.block(targetV.size(), 0, th.size(), targetV.size()) * dir
           + 2 * wM * M_theta * th
           + 2 * wL * L * th
           + wP * qg;
  };

  double energy = distance(theta);
  std::cout << "Initial SPN energy: " << energy << " " << energy <<"\t distance: "<<(x - xTarget).dot(masses.cwiseProduct(x - xTarget))<< std::endl;

  LUSolver solver;

  for(int i = 0; i < max_iters; ++i)
  {
    double f = distance(theta);
    Eigen::VectorXd g = distanceGrad(theta);

    Eigen::VectorXd b(2 * x.size() - 2 * fixedIdx.size() + theta.size());
    b.setZero();
    b.segment(x.size() - fixedIdx.size(), theta.size()) = -g;

    // penalty Hessian on theta
    qH = penaltyFunc.eval_hessian(theta);

    // rebuild HGN with extra theta-theta term from penalty
    HGN = buildHGN(2 * masses, P,
                   2 * wM * M_theta + 2 * wL * L + wP * qH,
                   H);

    if(i == 0)
      solver.compute(HGN);
    else
      solver.factorize(HGN);

    if(solver.info() != Eigen::Success)
    {
      std::cout << "Solver error\n";
      return targetV;
    }

    Eigen::VectorXd d = solver.solve(b);

    Eigen::VectorXd deltaTheta = d.segment(x.size() - fixedIdx.size(), theta.size());
    Eigen::VectorXd deltaX     = d.segment(0, x.size() - fixedIdx.size());
    deltaX = P.transpose() * deltaX;

    // line search
    Eigen::VectorXd x_old = x;
    double s = lineSearch(theta, deltaTheta, f, g, distance,
                          [&](double s) { x = x_old + s * deltaX; });

    if(s < 0)
    {
      // lineSearch's 32 shrinks each call eval() which runs forward newton
      // and can drag x into a different equilibrium basin.  When the search
      // ultimately fails, x is left in a contaminated state (not at
      // equilibrium for the current theta).  Revert x to the last-good
      // equilibrium (x_old) so the subsequent Final SPN distance() call
      // converges in the correct basin instead of jumping out to a wrong
      // local minimum.
      x = x_old;
      std::cout << "Line search failed (x reverted to last good state)\n";
      break;
    }

    theta += s * deltaTheta;

    std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaTheta, g)
              << "\tSPN energy: " << distance(theta)
              << "\tDistance: " << (x - xTarget).dot(masses.cwiseProduct(x - xTarget))
              << "\tStep size: " << s << std::endl;

    if(TinyAD::newton_decrement(deltaTheta, g) < lim || solver.info() != Eigen::Success)
      break;

    callback(x);
  }

  // Force one final forward-sim convergence on the reverted x so the
  // returned Vr reflects the *actual* stable elastic equilibrium for the
  // current (theta1, theta2) — not the half-converged state of iter N.
  // This makes OptLam-exit distance == OptKap-entry distance (no jump
  // between stages).  May reveal that the last iter's printed "Distance"
  // was misleading (forward sim wasn't fully converged), but that's
  // physically honest.
  const double final_energy = distance(theta);

  final_distance   = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
  final_spn_energy = final_energy;
  final_self_reg   = wM * theta.dot(M_theta * theta) + wL * theta.dot(L * theta);

  Eigen::MatrixXd V(targetV.rows(), 3);
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      V(i, j) = x(3 * i + j);

  theta1.fromVector(theta);
  return V;
}


// ===========================================================================
// MGDA variants (see docs/grayscalemorph/inverse_mgda.md)
//
// Replaces the single-objective wP-weighted penalty loop with a per-iteration
// Multiple-Gradient Descent step:
//   d_F = -H_F^{-1} g_F       (Newton step on SPN energy F, from KKT with wP=0)
//   d_P = snap(theta) - theta (Newton step on hard-min penalty, closed form)
//   alpha  = mgda_alpha(d_F, d_P)
//   d      = alpha * d_F + (1-alpha) * d_P
//   step s = two-objective Armijo (lineSearchMulti) on F and Phi
// Terminates on Pareto-critical norm < lim or max_iters.
// ===========================================================================

namespace {

// Per-face nearest-candidate snap (Voronoi projection in 1D).
inline double snap_to_candidate(double t, const std::vector<double>& cands)
{
  double best = cands.front();
  double best_d = (t - best) * (t - best);
  for(size_t i = 1; i < cands.size(); ++i)
  {
    const double d = (t - cands[i]) * (t - cands[i]);
    if(d < best_d) { best_d = d; best = cands[i]; }
  }
  return best;
}

inline Eigen::VectorXd snap_vector(const Eigen::VectorXd& th,
                                   const std::vector<double>& cands)
{
  Eigen::VectorXd s(th.size());
  for(int i = 0; i < th.size(); ++i)
    s(i) = snap_to_candidate(th(i), cands);
  return s;
}

// Efrati elastic-strain-energy distance² (per face) between current
// (lambda, kappa) and feasible pair j:
//   d²_j = lambda_bar_j² * (E/(1-nu)) * [ (h/4)*(lambda² - lambda_bar_j²)²
//                                        + (h³/12)*(kappa - kappa_bar_j)² ]
// self_is_kappa selects which variable `theta` represents.  Returns the
// index of the nearest candidate (Voronoi cell j*) for this face.
inline int nearest_efrati_index(double theta_self,
                                double other_const_f,
                                const std::vector<double>& cand_self,
                                const std::vector<double>& cand_other,
                                double E, double nu, double h,
                                bool self_is_kappa)
{
  const double C_pre  = E / (1.0 - nu);
  const double C_str  = h / 4.0;
  const double C_bend = (h * h * h) / 12.0;
  int best = 0; double best_d2 = std::numeric_limits<double>::infinity();
  for(size_t j = 0; j < cand_self.size(); ++j)
  {
    const double lam_bar = self_is_kappa ? cand_other[j] : cand_self[j];
    const double kap_bar = self_is_kappa ? cand_self[j]  : cand_other[j];
    const double lam = self_is_kappa ? other_const_f : theta_self;
    const double kap = self_is_kappa ? theta_self    : other_const_f;
    const double lam_str  = lam * lam - lam_bar * lam_bar;
    const double kap_bend = kap - kap_bar;
    const double d2 = lam_bar * lam_bar * C_pre
                    * (C_str  * lam_str  * lam_str
                     + C_bend * kap_bend * kap_bend);
    if(d2 < best_d2) { best_d2 = d2; best = static_cast<int>(j); }
  }
  return best;
}

// d_P for OptKap (self = kappa).  H is constant in cell j*, so the Newton
// step reduces to a snap: d_P_f = kappa_bar_{j*} - kappa_f.
inline Eigen::VectorXd dP_efrati_optkap(const Eigen::VectorXd& kappa,
                                        const std::vector<double>& cand_kap,
                                        const std::vector<double>& cand_lam,
                                        const std::vector<double>& lambda_const,
                                        double E, double nu, double h)
{
  Eigen::VectorXd d(kappa.size());
  for(int f = 0; f < kappa.size(); ++f)
  {
    const int j = nearest_efrati_index(kappa(f), lambda_const[f],
                                       cand_kap, cand_lam, E, nu, h,
                                       /*self_is_kappa=*/true);
    d(f) = cand_kap[j] - kappa(f);
  }
  return d;
}

// d_P for OptLam (self = lambda).  In cell j*:
//   g = lambda_bar² * (E h/(1-nu)) * lambda * (lambda² - lambda_bar²)
//   H = lambda_bar² * (E h/(1-nu)) * (3 lambda² - lambda_bar²)
//   d = -g / H
// If H is too small / non-positive (lambda << lambda_bar/sqrt(3)), fall
// back to a simple snap d = lambda_bar - lambda to avoid divide-by-zero
// and over-shoot.
inline Eigen::VectorXd dP_efrati_optlam(const Eigen::VectorXd& lambda,
                                        const std::vector<double>& cand_lam,
                                        const std::vector<double>& cand_kap,
                                        const std::vector<double>& kappa_const,
                                        double E, double nu, double h)
{
  const double H_guard = 1e-12;
  Eigen::VectorXd d(lambda.size());
  for(int f = 0; f < lambda.size(); ++f)
  {
    const int j = nearest_efrati_index(lambda(f), kappa_const[f],
                                       cand_lam, cand_kap, E, nu, h,
                                       /*self_is_kappa=*/false);
    const double lb = cand_lam[j];
    const double l  = lambda(f);
    const double lb_sq = lb * lb;
    const double l_sq  = l * l;
    const double H_factor = 3.0 * l_sq - lb_sq;
    if(std::abs(H_factor) < H_guard)
      d(f) = lb - l;  // fallback
    else
      d(f) = -l * (l_sq - lb_sq) / H_factor;
  }
  return d;
}

// 2D Euclidean joint snap (no physics weighting).  Penalty per face:
//   d²_j = (theta_self - cand_self[j])² + (other_const_f - cand_other[j])²
// Hessian wrt theta_self is the constant (2β/nF), so Newton step is the
// simple snap d_P = cand_self[j*] - theta_self with j* = arg min d²_j.
// Works for both OptKap and OptLam (symmetric formula).
inline int nearest_joint2d_index(double theta_self,
                                 double other_const_f,
                                 const std::vector<double>& cand_self,
                                 const std::vector<double>& cand_other)
{
  int best = 0; double best_d2 = std::numeric_limits<double>::infinity();
  for(size_t j = 0; j < cand_self.size(); ++j)
  {
    const double ds = theta_self    - cand_self[j];
    const double dot = other_const_f - cand_other[j];
    const double d2 = ds * ds + dot * dot;
    if(d2 < best_d2) { best_d2 = d2; best = static_cast<int>(j); }
  }
  return best;
}

inline Eigen::VectorXd dP_joint2d(const Eigen::VectorXd& theta,
                                  const std::vector<double>& cand_self,
                                  const std::vector<double>& cand_other,
                                  const std::vector<double>& other_const)
{
  Eigen::VectorXd d(theta.size());
  for(int f = 0; f < theta.size(); ++f)
  {
    const int j = nearest_joint2d_index(theta(f), other_const[f], cand_self, cand_other);
    d(f) = cand_self[j] - theta(f);
  }
  return d;
}

} // anonymous


Eigen::MatrixXd sparse_gauss_newton_FixLam_OptKap_MGDA(
    IntrinsicGeometryInterface& geometry,
    const Eigen::MatrixXd& targetV,
    const Eigen::MatrixXd& initV,
    const FaceData<Eigen::Matrix2d>& MrInv,
    FaceData<double>& theta1,    // lambda, fixed
    FaceData<double>& theta2,    // kappa, optimised
    const Eigen::VectorXd& masses,
    double other_reg,
    const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
    const TinyAD::ScalarFunction<1, double, Eigen::Index>& penaltyFunc,
    const std::vector<double>& candidate_vals,    // 1D candidates for the optimised variable
    double betaP,
    const std::vector<int>& fixedIdx,
    int max_iters,
    double lim,
    double wM,
    double wL,
    double E,
    double nu,
    double h,
    double w_s,
    double w_b,
    const std::vector<int>& ref_faces,
    double& final_distance,
    double& final_spn_energy,
    double& final_self_reg,
    double& final_penalty,
    double& final_pareto_norm,
    const std::function<void(const Eigen::VectorXd&)>& callback,
    const SgnIterCallback& iter_cb,
    const std::vector<double>& cand_other,
    const std::vector<double>& other_const,
    double alpha_override)
{
  geometry.requireFaceAreas();
  geometry.requireVertexIndices();

  SurfaceMesh& mesh = geometry.mesh;
  const size_t nF = mesh.nFaces();

  // Per-face regulariser matrices for theta2 (kappa).
  Eigen::SparseMatrix<double> M_theta(nF, nF);
  M_theta.reserve(nF);
  {
    size_t iF = 0;
    for(Face f : mesh.faces())
    {
      M_theta.insert(iF, iF) = 0.5 / MrInv[f].determinant();
      ++iF;
    }
  }

  Eigen::SparseMatrix<double> L(nF, nF);
  {
    FaceData<size_t> faceIdx(mesh);
    size_t cnt = 0;
    for(Face f : mesh.faces()) faceIdx[f] = cnt++;
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(mesh.nEdges() * 4);
    std::vector<double> diag(nF, 0.0);
    for(Edge e : mesh.edges())
    {
      if(e.isBoundary()) continue;
      Halfedge he = e.halfedge();
      size_t i = faceIdx[he.face()];
      size_t j = faceIdx[he.twin().face()];
      trips.emplace_back((int)i, (int)j, -1.0);
      trips.emplace_back((int)j, (int)i, -1.0);
      diag[i] += 1.0; diag[j] += 1.0;
    }
    for(size_t i = 0; i < nF; ++i)
      trips.emplace_back((int)i, (int)i, diag[i]);
    L.setFromTriplets(trips.begin(), trips.end());
  }

  Eigen::VectorXd theta = theta2.toVector();
  Eigen::VectorXd xTarget(targetV.size());
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      xTarget(3 * i + j) = targetV(i, j);
  Eigen::VectorXd x(initV.size());
  for(int i = 0; i < initV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      x(3 * i + j) = initV(i, j);

  LLTSolver adjointSolver;

  // F = distance + self_reg + other_reg (no penalty).  Runs forward newton
  // to drive x to equilibrium for the trial theta.
  auto F_eval = [&](const Eigen::VectorXd& th) {
    theta2.fromVector(th);
    auto simFunc = simulationFunction(geometry, MrInv, theta1, theta2, E, nu, h, w_s, w_b, ref_faces);
    newton(x, simFunc, adjointSolver, 100, lim, false, fixedIdx);
    return (x - xTarget).dot(masses.cwiseProduct(x - xTarget))
           + wM * th.dot(M_theta * th)
           + wL * th.dot(L * th)
           + other_reg;
  };

  // Phi = TinyAD penalty (no forward sim).  Cheap, no x update.
  auto Phi_eval = [&](const Eigen::VectorXd& th) {
    return penaltyFunc.eval(th);
  };

  Eigen::SparseMatrix<double> P = projectionMatrix(fixedIdx, x.size());

  // grad F via adjoint method (same logic as _Penalty's distanceGrad
  // without the wP*qg term).
  Eigen::SparseMatrix<double> H;
  auto F_grad = [&](const Eigen::VectorXd& th) -> Eigen::VectorXd {
    Eigen::VectorXd X(targetV.size() + th.size());
    X.head(targetV.size()) = x;
    X.tail(th.size()) = th;
    H = adjointFunc.eval_hessian(X);
    for(int j = 0; j < targetV.size(); ++j)
      H.coeffRef(j, j) += 1e-10;
    Eigen::SparseMatrix<double> A = (P * H.block(0, 0, targetV.size(), targetV.size()) * P.transpose()).eval();
    adjointSolver.factorize(A);
    if(adjointSolver.info() != Eigen::Success)
    {
      auto [f, g, A_proj] = adjointFunc.eval_with_hessian_proj(X);
      A_proj = (P * A_proj.block(0, 0, targetV.size(), targetV.size()) * P.transpose()).eval();
      A = 0.9 * A + 0.1 * A_proj;
      adjointSolver.factorize(A);
      if(adjointSolver.info() != Eigen::Success)
        adjointSolver.factorize(A_proj);
    }
    Eigen::VectorXd b = P * masses.cwiseProduct(x - xTarget);
    Eigen::VectorXd dir = adjointSolver.solve(b);
    if(adjointSolver.info() != Eigen::Success)
      std::cout << "Solver error (F_grad)\n";
    dir = P.transpose() * dir;
    return -2 * H.block(targetV.size(), 0, th.size(), targetV.size()) * dir
           + 2 * wM * M_theta * th
           + 2 * wL * L * th;
  };

  auto Phi_grad = [&](const Eigen::VectorXd& th) -> Eigen::VectorXd {
    auto [qf, qg] = penaltyFunc.eval_with_gradient(th);
    return qg;
  };

  // Initial state.
  double f_F = F_eval(theta);
  double f_P = Phi_eval(theta);
  std::cout << "Initial F=" << f_F << "\tPhi=" << f_P
            << "\tdistance=" << (x - xTarget).dot(masses.cwiseProduct(x - xTarget))
            << std::endl;

  LUSolver solver;
  Eigen::VectorXd deltaTheta_F;   // last accepted d_F (theta part)
  Eigen::VectorXd deltaX_F;       // last accepted d_F (x part)
  double last_alpha = 1.0;
  double last_pareto = 0.0;

  for(int i = 0; i < max_iters; ++i)
  {
    // Refresh grads / Hessian.
    const Eigen::VectorXd g_F = F_grad(theta);  // also updates H
    const Eigen::VectorXd g_P = Phi_grad(theta);

    // Newton direction for F via KKT (wP = 0 in the theta-theta block).
    Eigen::VectorXd b(2 * x.size() - 2 * fixedIdx.size() + theta.size());
    b.setZero();
    b.segment(x.size() - fixedIdx.size(), theta.size()) = -g_F;

    Eigen::SparseMatrix<double> HGN = buildHGN(2 * masses, P, 2 * wM * M_theta + 2 * wL * L, H);
    if(i == 0)
      solver.compute(HGN);
    else
      solver.factorize(HGN);
    if(solver.info() != Eigen::Success)
    {
      std::cout << "Solver error (HGN factorize)\n";
      break;
    }

    const Eigen::VectorXd d_kkt = solver.solve(b);
    deltaTheta_F = d_kkt.segment(x.size() - fixedIdx.size(), theta.size());
    deltaX_F     = P.transpose() * d_kkt.segment(0, x.size() - fixedIdx.size());

    // Newton direction for Phi:
    //  - cand_other empty -> 1D independent hard-min penalty (closed form,
    //    snap to nearest 1D candidate on candidate_vals)
    //  - cand_other non-empty -> 2D Euclidean joint penalty (closed form,
    //    snap to candidate_vals[j*] where j* picks the (cand_self,cand_other)
    //    pair nearest to (theta, other_const_f) in plain 2D)
    Eigen::VectorXd d_P;
    if (cand_other.empty()) {
      const Eigen::VectorXd snap_theta = snap_vector(theta, candidate_vals);
      d_P = snap_theta - theta;
    } else {
      d_P = dP_joint2d(theta, candidate_vals, cand_other, other_const);
    }

    // MGDA combine (Newton-step level).
    // Closed-form MGDA alpha unless the caller imposed a stage-wise override.
    const double alpha = (alpha_override >= 0.0)
                       ? std::clamp(alpha_override, 0.0, 1.0)
                       : mgda_alpha(deltaTheta_F, d_P);
    const Eigen::VectorXd d = alpha * deltaTheta_F + (1.0 - alpha) * d_P;
    last_alpha = alpha;

    // Pareto-critical norm (first-order, separate alpha on gradients).
    const double alpha_g = mgda_alpha(g_F, g_P);
    const double pareto_norm = (alpha_g * g_F + (1.0 - alpha_g) * g_P).norm();
    last_pareto = pareto_norm;

    if(d.squaredNorm() < lim * lim || pareto_norm < lim)
    {
      std::cout << "iter " << i
                << "  alpha=" << alpha
                << "  ||d||=" << d.norm()
                << "  pareto=" << pareto_norm
                << "  F=" << f_F << "  Phi=" << f_P
                << "  [Pareto critical]\n";
      break;
    }

    // Two-objective Armijo line search.  Warm-start x with alpha * deltaX_F
    // (d_P has no associated x-delta — forward newton inside F_eval re-equilibrates).
    Eigen::VectorXd x_old = x;
    const Eigen::VectorXd dX = alpha * deltaX_F;
    double s = lineSearchMulti(
        theta, d,
        f_F, g_F,
        f_P, g_P,
        F_eval, Phi_eval,
        [&](double s) { x = x_old + s * dX; });

    if(s < 0)
    {
      x = x_old;
      std::cout << "Two-objective line search failed (x reverted)\n";
      break;
    }

    theta += s * d;
    f_F = F_eval(theta);   // already re-equilibrated x for new theta
    f_P = Phi_eval(theta);

    const double iter_dist = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    const double iter_self_reg = wM * theta.dot(M_theta * theta) + wL * theta.dot(L * theta);
    std::cout << "iter " << i
              << "  alpha=" << alpha
              << "  ||d||=" << d.norm()
              << "  pareto=" << pareto_norm
              << "  F=" << f_F
              << "  Phi=" << f_P
              << "  dist=" << iter_dist
              << "  step=" << s << "\n";
    iter_cb(i, x, f_F, iter_dist, iter_self_reg, /*penalty=*/f_P);

    callback(x);
  }

  // Force one final forward-sim convergence on x (mirrors _Penalty postlude).
  const double final_F = F_eval(theta);
  final_distance   = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
  final_spn_energy = final_F;
  final_self_reg   = wM * theta.dot(M_theta * theta) + wL * theta.dot(L * theta);
  final_penalty    = Phi_eval(theta);
  final_pareto_norm = last_pareto;
  (void)last_alpha;  // diagnostic only, not returned

  Eigen::MatrixXd V(targetV.rows(), 3);
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      V(i, j) = x(3 * i + j);

  theta2.fromVector(theta);
  return V;
}


Eigen::MatrixXd sparse_gauss_newton_FixKap_OptLam_MGDA(
    IntrinsicGeometryInterface& geometry,
    const Eigen::MatrixXd& targetV,
    const Eigen::MatrixXd& initV,
    const FaceData<Eigen::Matrix2d>& MrInv,
    FaceData<double>& theta1,    // lambda, optimised
    FaceData<double>& theta2,    // kappa, fixed
    const Eigen::VectorXd& masses,
    double other_reg,
    const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
    const TinyAD::ScalarFunction<1, double, Eigen::Index>& penaltyFunc,
    const std::vector<double>& candidate_vals,    // 1D candidates for the optimised variable
    double betaP,
    const std::vector<int>& fixedIdx,
    int max_iters,
    double lim,
    double wM,
    double wL,
    double E,
    double nu,
    double h,
    double w_s,
    double w_b,
    const std::vector<int>& ref_faces,
    double& final_distance,
    double& final_spn_energy,
    double& final_self_reg,
    double& final_penalty,
    double& final_pareto_norm,
    const std::function<void(const Eigen::VectorXd&)>& callback,
    const SgnIterCallback& iter_cb,
    const std::vector<double>& cand_other,
    const std::vector<double>& other_const,
    double alpha_override)
{
  geometry.requireFaceAreas();
  geometry.requireVertexIndices();

  SurfaceMesh& mesh = geometry.mesh;
  const int nF = static_cast<int>(mesh.nFaces());

  // Face area mass matrix (lambda regulariser).
  Eigen::SparseMatrix<double> M_theta(nF, nF);
  M_theta.reserve(nF);
  {
    int iF = 0;
    for(Face f : mesh.faces())
    {
      M_theta.insert(iF, iF) = geometry.faceAreas[f];
      ++iF;
    }
  }

  Eigen::SparseMatrix<double> L(nF, nF);
  {
    FaceData<int> faceIdx(mesh);
    int cnt = 0;
    for(Face f : mesh.faces()) faceIdx[f] = cnt++;
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(mesh.nEdges() * 4);
    std::vector<double> diag(nF, 0.0);
    for(Edge e : mesh.edges())
    {
      if(e.isBoundary()) continue;
      Halfedge he = e.halfedge();
      int i = faceIdx[he.face()];
      int j = faceIdx[he.twin().face()];
      trips.emplace_back(i, j, -1.0);
      trips.emplace_back(j, i, -1.0);
      diag[i] += 1.0; diag[j] += 1.0;
    }
    for(int i = 0; i < nF; ++i)
      trips.emplace_back(i, i, diag[i]);
    L.setFromTriplets(trips.begin(), trips.end());
  }

  Eigen::VectorXd theta = theta1.toVector();
  Eigen::VectorXd xTarget(targetV.size());
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      xTarget(3 * i + j) = targetV(i, j);
  Eigen::VectorXd x(initV.size());
  for(int i = 0; i < initV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      x(3 * i + j) = initV(i, j);

  LLTSolver adjointSolver;

  auto F_eval = [&](const Eigen::VectorXd& th) {
    theta1.fromVector(th);
    auto simFunc = simulationFunction(geometry, MrInv, theta1, theta2, E, nu, h, w_s, w_b, ref_faces);
    newton(x, simFunc, adjointSolver, 100, lim, false, fixedIdx);
    return (x - xTarget).dot(masses.cwiseProduct(x - xTarget))
           + wM * th.dot(M_theta * th)
           + wL * th.dot(L * th)
           + other_reg;
  };

  auto Phi_eval = [&](const Eigen::VectorXd& th) {
    return penaltyFunc.eval(th);
  };

  Eigen::SparseMatrix<double> P = projectionMatrix(fixedIdx, x.size());
  Eigen::SparseMatrix<double> H;

  auto F_grad = [&](const Eigen::VectorXd& th) -> Eigen::VectorXd {
    Eigen::VectorXd X(targetV.size() + th.size());
    X.head(targetV.size()) = x;
    X.tail(th.size()) = th;
    H = adjointFunc.eval_hessian(X);
    for(int j = 0; j < targetV.size(); ++j)
      H.coeffRef(j, j) += 1e-10;
    Eigen::SparseMatrix<double> A = (P * H.block(0, 0, targetV.size(), targetV.size()) * P.transpose()).eval();
    adjointSolver.factorize(A);
    if(adjointSolver.info() != Eigen::Success)
    {
      auto [f, g, A_proj] = adjointFunc.eval_with_hessian_proj(X);
      A_proj = (P * A_proj.block(0, 0, targetV.size(), targetV.size()) * P.transpose()).eval();
      A = 0.9 * A + 0.1 * A_proj;
      adjointSolver.factorize(A);
      if(adjointSolver.info() != Eigen::Success)
        adjointSolver.factorize(A_proj);
    }
    Eigen::VectorXd b = P * masses.cwiseProduct(x - xTarget);
    Eigen::VectorXd dir = adjointSolver.solve(b);
    if(adjointSolver.info() != Eigen::Success)
      std::cout << "Solver error (F_grad)\n";
    dir = P.transpose() * dir;
    return -2 * H.block(targetV.size(), 0, th.size(), targetV.size()) * dir
           + 2 * wM * M_theta * th
           + 2 * wL * L * th;
  };

  auto Phi_grad = [&](const Eigen::VectorXd& th) -> Eigen::VectorXd {
    auto [qf, qg] = penaltyFunc.eval_with_gradient(th);
    return qg;
  };

  double f_F = F_eval(theta);
  double f_P = Phi_eval(theta);
  std::cout << "Initial F=" << f_F << "\tPhi=" << f_P
            << "\tdistance=" << (x - xTarget).dot(masses.cwiseProduct(x - xTarget))
            << std::endl;

  LUSolver solver;
  Eigen::VectorXd deltaTheta_F, deltaX_F;
  double last_alpha = 1.0;
  double last_pareto = 0.0;

  for(int i = 0; i < max_iters; ++i)
  {
    const Eigen::VectorXd g_F = F_grad(theta);  // updates H
    const Eigen::VectorXd g_P = Phi_grad(theta);

    Eigen::VectorXd b(2 * x.size() - 2 * fixedIdx.size() + theta.size());
    b.setZero();
    b.segment(x.size() - fixedIdx.size(), theta.size()) = -g_F;

    Eigen::SparseMatrix<double> HGN = buildHGN(2 * masses, P, 2 * wM * M_theta + 2 * wL * L, H);
    if(i == 0)
      solver.compute(HGN);
    else
      solver.factorize(HGN);
    if(solver.info() != Eigen::Success)
    {
      std::cout << "Solver error (HGN factorize)\n";
      break;
    }

    const Eigen::VectorXd d_kkt = solver.solve(b);
    deltaTheta_F = d_kkt.segment(x.size() - fixedIdx.size(), theta.size());
    deltaX_F     = P.transpose() * d_kkt.segment(0, x.size() - fixedIdx.size());

    // Newton direction for Phi:
    //  - cand_other empty -> 1D independent hard-min penalty (closed form,
    //    snap to nearest 1D candidate on candidate_vals)
    //  - cand_other non-empty -> 2D Euclidean joint penalty (closed form,
    //    snap to candidate_vals[j*] where j* picks the (cand_self,cand_other)
    //    pair nearest to (theta, other_const_f) in plain 2D)
    Eigen::VectorXd d_P;
    if (cand_other.empty()) {
      const Eigen::VectorXd snap_theta = snap_vector(theta, candidate_vals);
      d_P = snap_theta - theta;
    } else {
      d_P = dP_joint2d(theta, candidate_vals, cand_other, other_const);
    }

    // Closed-form MGDA alpha unless the caller imposed a stage-wise override.
    const double alpha = (alpha_override >= 0.0)
                       ? std::clamp(alpha_override, 0.0, 1.0)
                       : mgda_alpha(deltaTheta_F, d_P);
    const Eigen::VectorXd d = alpha * deltaTheta_F + (1.0 - alpha) * d_P;
    last_alpha = alpha;

    const double alpha_g = mgda_alpha(g_F, g_P);
    const double pareto_norm = (alpha_g * g_F + (1.0 - alpha_g) * g_P).norm();
    last_pareto = pareto_norm;

    if(d.squaredNorm() < lim * lim || pareto_norm < lim)
    {
      std::cout << "iter " << i
                << "  alpha=" << alpha
                << "  ||d||=" << d.norm()
                << "  pareto=" << pareto_norm
                << "  F=" << f_F << "  Phi=" << f_P
                << "  [Pareto critical]\n";
      break;
    }

    Eigen::VectorXd x_old = x;
    const Eigen::VectorXd dX = alpha * deltaX_F;
    double s = lineSearchMulti(
        theta, d,
        f_F, g_F,
        f_P, g_P,
        F_eval, Phi_eval,
        [&](double s) { x = x_old + s * dX; });

    if(s < 0)
    {
      x = x_old;
      std::cout << "Two-objective line search failed (x reverted)\n";
      break;
    }

    theta += s * d;
    f_F = F_eval(theta);
    f_P = Phi_eval(theta);

    const double iter_dist = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    const double iter_self_reg = wM * theta.dot(M_theta * theta) + wL * theta.dot(L * theta);
    std::cout << "iter " << i
              << "  alpha=" << alpha
              << "  ||d||=" << d.norm()
              << "  pareto=" << pareto_norm
              << "  F=" << f_F
              << "  Phi=" << f_P
              << "  dist=" << iter_dist
              << "  step=" << s << "\n";
    iter_cb(i, x, f_F, iter_dist, iter_self_reg, /*penalty=*/f_P);

    callback(x);
  }

  const double final_F = F_eval(theta);
  final_distance   = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
  final_spn_energy = final_F;
  final_self_reg   = wM * theta.dot(M_theta * theta) + wL * theta.dot(L * theta);
  final_penalty    = Phi_eval(theta);
  final_pareto_norm = last_pareto;
  (void)last_alpha;
  (void)betaP;  // currently used only via TinyAD penaltyFunc; reserved for explicit Hessian if needed

  Eigen::MatrixXd V(targetV.rows(), 3);
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      V(i, j) = x(3 * i + j);

  theta1.fromVector(theta);
  return V;
}


// ===========================================================================
//  SGN OptP: variable = 2D parameterisation P, material (lambda, kappa) const
// ===========================================================================
Eigen::MatrixXd sparse_gauss_newton_FixMaterial_OptP(
    IntrinsicGeometryInterface& geometry,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& targetV,
    const Eigen::MatrixXd& initV,
    Eigen::MatrixXd& P_io,
    const FaceData<double>& lambda_pf,
    const FaceData<double>& kappa_pf,
    const Eigen::VectorXd& masses,
    const Eigen::SparseMatrix<double>& M_P,
    const Eigen::SparseMatrix<double>& L_P,
    const Eigen::MatrixXd& P_anchor,
    double other_reg,
    const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
    const std::vector<int>& fixedIdx,
    int max_iters,
    double lim,
    double wM_P,
    double wL_P,
    double E,
    double nu,
    double h,
    double w_s,
    double w_b,
    const std::vector<int>& ref_faces,
    double& final_distance,
    double& final_spn_energy,
    double& final_self_reg,
    const SgnIterCallback& iter_cb,
    const std::function<void(const Eigen::VectorXd&)>& callback)
{
  geometry.requireFaceAreas();
  geometry.requireVertexIndices();

  SurfaceMesh& mesh = geometry.mesh;
  const size_t nV = mesh.nVertices();
  const Eigen::Index P_size = static_cast<Eigen::Index>(2 * nV);

  Eigen::VectorXd xTarget(targetV.size());
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      xTarget(3 * i + j) = targetV(i, j);
  Eigen::VectorXd x(initV.size());
  for(int i = 0; i < initV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      x(3 * i + j) = initV(i, j);

  Eigen::VectorXd P_vec(P_size);
  for(int i = 0; i < (int)nV; ++i) {
    P_vec(2 * i + 0) = P_io(i, 0);
    P_vec(2 * i + 1) = P_io(i, 1);
  }
  Eigen::VectorXd P_anchor_vec(P_size);
  for(int i = 0; i < (int)nV; ++i) {
    P_anchor_vec(2 * i + 0) = P_anchor(i, 0);
    P_anchor_vec(2 * i + 1) = P_anchor(i, 1);
  }

  auto unpack_P = [&](const Eigen::VectorXd& Pv) {
    Eigen::MatrixXd Pm(nV, 2);
    for(int i = 0; i < (int)nV; ++i) {
      Pm(i, 0) = Pv(2 * i + 0);
      Pm(i, 1) = Pv(2 * i + 1);
    }
    return Pm;
  };

  LLTSolver adjointSolver;

  auto distance = [&](const Eigen::VectorXd& Pv) -> double {
    Eigen::MatrixXd Pm = unpack_P(Pv);
    // Pre-check: every face's Mr = [P1-P0, P2-P0] must have det > 0.
    for(int fi = 0; fi < F.rows(); ++fi) {
      const Eigen::Vector2d e1 = Pm.row(F(fi, 1)) - Pm.row(F(fi, 0));
      const Eigen::Vector2d e2 = Pm.row(F(fi, 2)) - Pm.row(F(fi, 0));
      const double det = e1.x() * e2.y() - e1.y() * e2.x();
      if (!std::isfinite(det) || det < 1e-10)
        return std::numeric_limits<double>::infinity();
    }
    P_io = Pm;
    FaceData<Eigen::Matrix2d> MrInv_curr = precomputeMrInv(
        *dynamic_cast<ManifoldSurfaceMesh*>(&mesh), Pm, F);
    auto simFunc = simulationFunction(geometry, MrInv_curr, lambda_pf, kappa_pf,
                                      E, nu, h, w_s, w_b, ref_faces);
    newton(x, simFunc, adjointSolver, 100, lim, false, fixedIdx);

    const Eigen::VectorXd Pd = Pv - P_anchor_vec;
    const double dist_term      = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    if (!std::isfinite(dist_term))
      return std::numeric_limits<double>::infinity();
    const double anchor_reg     = wM_P * Pd.dot(M_P * Pd);
    const double smoothness_reg = wL_P * Pv.dot(L_P * Pv);
    return dist_term + anchor_reg + smoothness_reg + other_reg;
  };

  Eigen::SparseMatrix<double> P_proj = projectionMatrix(fixedIdx, x.size());
  Eigen::SparseMatrix<double> M_theta_kkt = 2.0 * wM_P * M_P + 2.0 * wL_P * L_P;

  Eigen::VectorXd X(targetV.size() + P_size);
  X.head(targetV.size()) = x;
  X.tail(P_size) = P_vec;
  Eigen::SparseMatrix<double> H = adjointFunc.eval_hessian(X);
  Eigen::SparseMatrix<double> HGN = buildHGN(2 * masses, P_proj, M_theta_kkt, H);

  auto distanceGrad = [&](const Eigen::VectorXd& Pv) -> Eigen::VectorXd {
    Eigen::VectorXd X(targetV.size() + Pv.size());
    X.head(targetV.size()) = x;
    X.tail(Pv.size()) = Pv;
    H = adjointFunc.eval_hessian(X);

    for(int j = 0; j < targetV.size(); ++j)
      H.coeffRef(j, j) += 1e-10;

    Eigen::SparseMatrix<double> A =
        (P_proj * H.block(0, 0, targetV.size(), targetV.size()) * P_proj.transpose()).eval();

    adjointSolver.factorize(A);
    if(adjointSolver.info() != Eigen::Success) {
      auto [f, g, A_proj] = adjointFunc.eval_with_hessian_proj(X);
      A_proj = (P_proj * A_proj.block(0, 0, targetV.size(), targetV.size()) * P_proj.transpose()).eval();
      A = 0.9 * A + 0.1 * A_proj;
      adjointSolver.factorize(A);
      if(adjointSolver.info() != Eigen::Success)
        adjointSolver.factorize(A_proj);
    }

    Eigen::VectorXd b = P_proj * masses.cwiseProduct(x - xTarget);
    Eigen::VectorXd dir = adjointSolver.solve(b);
    if(adjointSolver.info() != Eigen::Success)
      std::cout << "Solver error\n";

    dir = P_proj.transpose() * dir;

    return -2 * H.block(targetV.size(), 0, Pv.size(), targetV.size()) * dir
         + 2 * wM_P * M_P * (Pv - P_anchor_vec)
         + 2 * wL_P * L_P * Pv;
  };

  double energy0 = distance(P_vec);
  std::cout << "Initial SPN energy (OptP): " << energy0
            << "\t distance: " << (x - xTarget).dot(masses.cwiseProduct(x - xTarget)) << std::endl;

  LUSolver solver;

  for(int i = 0; i < max_iters; ++i)
  {
    double f = distance(P_vec);
    Eigen::VectorXd g = distanceGrad(P_vec);

    Eigen::VectorXd b(2 * x.size() - 2 * fixedIdx.size() + P_vec.size());
    b.setZero();
    b.segment(x.size() - fixedIdx.size(), P_vec.size()) = -g;

    updateHGN(HGN, P_proj, H);

    if(i == 0) solver.compute(HGN);
    else        solver.factorize(HGN);

    if(solver.info() != Eigen::Success) {
      std::cout << "Solver error (OptP)\n";
      final_distance = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
      final_spn_energy = f;
      final_self_reg = wM_P * (P_vec - P_anchor_vec).dot(M_P * (P_vec - P_anchor_vec))
                     + wL_P * P_vec.dot(L_P * P_vec);
      Eigen::MatrixXd Vr(initV.rows(), 3);
      for(int v = 0; v < initV.rows(); ++v)
        for(int j = 0; j < 3; ++j)
          Vr(v, j) = x(3 * v + j);
      return Vr;
    }

    Eigen::VectorXd d = solver.solve(b);
    Eigen::VectorXd deltaP = d.segment(x.size() - fixedIdx.size(), P_vec.size());
    Eigen::VectorXd deltaX = d.segment(0, x.size() - fixedIdx.size());
    deltaX = P_proj.transpose() * deltaX;

    Eigen::VectorXd x_old = x;
    double s = lineSearch(P_vec, deltaP, f, g, distance, [&](double s){ x = x_old + s * deltaX; });
    if(s < 0) {
      x = x_old;
      std::cout << "Line search failed (OptP); P reverted.\n";
      break;
    }
    P_vec += s * deltaP;

    const double _iter_spn  = distance(P_vec);
    const double _iter_dist = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaP, g)
              << "\tSPN energy: " << _iter_spn
              << "\tDistance: " << _iter_dist
              << "\tStep size: " << s << std::endl;
    const Eigen::VectorXd Pd_iter = P_vec - P_anchor_vec;
    const double _iter_self_reg = wM_P * Pd_iter.dot(M_P * Pd_iter)
                                + wL_P * P_vec.dot(L_P * P_vec);
    iter_cb(i, x, _iter_spn, _iter_dist, _iter_self_reg, 0.0);

    if(TinyAD::newton_decrement(deltaP, g) < lim || solver.info() != Eigen::Success)
      break;

    callback(x);
  }

  P_io = unpack_P(P_vec);
  FaceData<Eigen::Matrix2d> MrInv_final = precomputeMrInv(
      *dynamic_cast<ManifoldSurfaceMesh*>(&mesh), P_io, F);
  auto simFunc_final = simulationFunction(geometry, MrInv_final, lambda_pf, kappa_pf,
                                          E, nu, h, w_s, w_b, ref_faces);
  newton(x, simFunc_final, adjointSolver, 100, lim, false, fixedIdx);

  final_distance   = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
  const Eigen::VectorXd Pd_final = P_vec - P_anchor_vec;
  final_self_reg   = wM_P * Pd_final.dot(M_P * Pd_final)
                   + wL_P * P_vec.dot(L_P * P_vec);
  final_spn_energy = final_distance + final_self_reg + other_reg;

  Eigen::MatrixXd Vr(initV.rows(), 3);
  for(int v = 0; v < initV.rows(); ++v)
    for(int j = 0; j < 3; ++j)
      Vr(v, j) = x(3 * v + j);
  return Vr;
}


// ---------------------------------------------------------------------------
// Explicit template instantiations for the newton<Func> overloads used outside
// this translation unit.  Previously lay1/lay2 lived in this file and pulled
// these instantiations in for free; after removing them we need to declare
// them explicitly so main.cpp / parameterization.cpp can link.
// ---------------------------------------------------------------------------
#include "functions.h"

template void newton<
    TinyAD::ScalarFunction<3, double, geometrycentral::surface::VertexRangeF::Etype>>(
    geometrycentral::surface::IntrinsicGeometryInterface& geometry,
    Eigen::MatrixXd& V,
    TinyAD::ScalarFunction<3, double, geometrycentral::surface::VertexRangeF::Etype>& func,
    int max_iters,
    double lim,
    bool verbose,
    const std::vector<int>& fixedIdx,
    const std::function<void(const Eigen::VectorXd&)>& callback);

template void newton<
    TinyAD::ScalarFunction<2, double, geometrycentral::surface::VertexRangeF::Etype>>(
    geometrycentral::surface::IntrinsicGeometryInterface& geometry,
    Eigen::MatrixXd& V,
    TinyAD::ScalarFunction<2, double, geometrycentral::surface::VertexRangeF::Etype>& func,
    int max_iters,
    double lim,
    bool verbose,
    const std::vector<int>& fixedIdx,
    const std::function<void(const Eigen::VectorXd&)>& callback);
