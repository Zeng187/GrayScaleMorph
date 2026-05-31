#include "newton.h"
#include "functions.h"
#include "parameterization.h"
#include "simulation_utils.h"
#include "morph_functions.hpp"
#include "solvers.h"
#include "timer.h"

#include <TinyAD/Utils/NewtonDecrement.hh>

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
                                    double theta_anchor,
                                    const M_Surface_LK& E_surface,
                                    double nu,
                                    double h,
                                    double w_s,
                                    double w_b,
                                    const std::vector<int>& ref_faces,
                                    double& final_distance,
                                    double& final_spn_energy,
                                    double& final_self_reg,
                                    const std::function<void(int, const Eigen::VectorXd&, double, double, double, double)>& iter_logger,
                                    const std::function<void(const Eigen::VectorXd&)>& callback)
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
  const Eigen::VectorXd anchor_vec = Eigen::VectorXd::Constant(theta.size(), theta_anchor);
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
    auto simFunc = simulationFunction(geometry, MrInv, theta1, theta2, E_surface,nu,h,w_s,w_b, ref_faces);
    newton(x, simFunc, adjointSolver, 100, lim, false, fixedIdx);

    // Unified SPN energy = distance + self regulariser + other-variable regulariser (constant in this stage)
    return (x - xTarget).dot(masses.cwiseProduct(x - xTarget)) + wM * (th - anchor_vec).dot(M_theta * (th - anchor_vec)) + wL * th.dot(L * th) + other_reg;
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

    return -2 * H.block(targetV.size(), 0, th.size(), targetV.size()) * dir + 2 * wM * M_theta * (th - anchor_vec) + 2 * wL * L * th;
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

    const double _iter_spn  = distance(theta);
    const double _iter_dist = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaTheta, g)
              << "\tSPN energy: " << _iter_spn
              << "\tDistance: " << _iter_dist
              << "\tStep size: " << s;
    const double _iter_self_reg = wM * (theta - anchor_vec).dot(M_theta * (theta - anchor_vec)) + wL * theta.dot(L * theta);
    iter_logger(i, x, _iter_spn, _iter_dist, _iter_self_reg, 0.0);
    if(TinyAD::newton_decrement(deltaTheta, g) < lim || solver.info() != Eigen::Success)
      break;

    callback(x);
  }

  // Force final forward-sim convergence — see other variants for rationale.
  const double final_energy = distance(theta);
  final_distance   = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
  final_spn_energy = final_energy;
  final_self_reg   = wM * (theta - anchor_vec).dot(M_theta * (theta - anchor_vec)) + wL * theta.dot(L * theta);

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
                                  double theta_anchor,
                                  const M_Surface_LK& E_surface,
                                  double nu,
                                  double h,
                                  double w_s,
                                  double w_b,
                                  const std::vector<int>& ref_faces,
                                  double& final_distance,
                                  double& final_spn_energy,
                                  double& final_self_reg,
                                  const std::function<void(int, const Eigen::VectorXd&, double, double, double, double)>& iter_logger,
                                  const std::function<void(const Eigen::VectorXd&)>& callback)
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
  const Eigen::VectorXd anchor_vec = Eigen::VectorXd::Constant(theta.size(), theta_anchor);

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
    auto simFunc = simulationFunction(geometry, MrInv, theta1, theta2, E_surface, nu, h, w_s, w_b, ref_faces);

    newton(x, simFunc, adjointSolver, 100, lim, false, fixedIdx);

    // Unified SPN energy = distance + self regulariser + other-variable regulariser (constant in this stage)
    return (x - xTarget).dot(masses.cwiseProduct(x - xTarget))
           + wM * (th - anchor_vec).dot(M_theta * (th - anchor_vec))
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
           + 2 * wM * M_theta * (th - anchor_vec)
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

    const double _iter_spn  = distance(theta);
    const double _iter_dist = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaTheta, g)
              << "\tSPN energy: " << _iter_spn
              << "\tDistance: " << _iter_dist
              << "\tStep size: " << s;
    const double _iter_self_reg = wM * (theta - anchor_vec).dot(M_theta * (theta - anchor_vec)) + wL * theta.dot(L * theta);
    iter_logger(i, x, _iter_spn, _iter_dist, _iter_self_reg, 0.0);

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
  final_self_reg   = wM * (theta - anchor_vec).dot(M_theta * (theta - anchor_vec)) + wL * theta.dot(L * theta);

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
                                    double theta_anchor,
                                    double wP,
                                    const M_Surface_LK& E_surface,
                                    double nu,
                                    double h,
                                    double w_s,
                                    double w_b,
                                    const std::vector<int>& ref_faces,
                                    double& final_distance,
                                    double& final_spn_energy,
                                    double& final_self_reg,
                                    const std::function<void(int, const Eigen::VectorXd&, double, double, double, double)>& iter_logger,
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
  const Eigen::VectorXd anchor_vec = Eigen::VectorXd::Constant(theta.size(), theta_anchor);
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
    auto simFunc = simulationFunction(geometry, MrInv, theta1, theta2, E_surface,nu,h,w_s,w_b, ref_faces);
    newton(x, simFunc, adjointSolver, 100, lim, false, fixedIdx);

    double qp = penaltyFunc.eval(th);
    // Unified SPN energy = distance + self regulariser + wP·penalty + other-variable regulariser (constant)
    return (x - xTarget).dot(masses.cwiseProduct(x - xTarget)) + wM * (th - anchor_vec).dot(M_theta * (th - anchor_vec)) + wL * th.dot(L * th) +
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
    return -2 * H.block(targetV.size(), 0, th.size(), targetV.size()) * dir + 2 * wM * M_theta * (th - anchor_vec) + 2 * wL * L * th + wP * qg;
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

    const double _iter_spn  = distance(theta);
    const double _iter_dist = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaTheta, g)
              << "\tSPN energy: " << _iter_spn
              << "\tDistance: " << _iter_dist
              << "\tStep size: " << s;
    const double _iter_self_reg = wM * (theta - anchor_vec).dot(M_theta * (theta - anchor_vec)) + wL * theta.dot(L * theta);
    const double _iter_penalty  = penaltyFunc.eval(theta);
    iter_logger(i, x, _iter_spn, _iter_dist, _iter_self_reg, _iter_penalty);
    if(TinyAD::newton_decrement(deltaTheta, g) < lim || solver.info() != Eigen::Success)
      break;

    callback(x);
  }

  // Force final forward-sim convergence — see other variants for rationale.
  const double final_energy = distance(theta);
  final_distance   = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
  final_spn_energy = final_energy;
  final_self_reg   = wM * (theta - anchor_vec).dot(M_theta * (theta - anchor_vec)) + wL * theta.dot(L * theta);

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
                                                          double theta_anchor,
                                                          double wP,
                                                          const M_Surface_LK& E_surface,
                                                          double nu,
                                                          double h,
                                                          double w_s,
                                                          double w_b,
                                                          const std::vector<int>& ref_faces,
                                                          double& final_distance,
                                                          double& final_spn_energy,
                                                          double& final_self_reg,
                                                          const std::function<void(int, const Eigen::VectorXd&, double, double, double, double)>& iter_logger,
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
  const Eigen::VectorXd anchor_vec = Eigen::VectorXd::Constant(theta.size(), theta_anchor);

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
    auto simFunc = simulationFunction(geometry, MrInv, theta1, theta2, E_surface, nu, h, w_s, w_b, ref_faces);
    newton(x, simFunc, adjointSolver, 100, lim, false, fixedIdx);

    double qp = penaltyFunc.eval(th);

    // Unified SPN energy = distance + self regulariser + wP·penalty + other-variable regulariser (constant)
    return (x - xTarget).dot(masses.cwiseProduct(x - xTarget))
           + wM * (th - anchor_vec).dot(M_theta * (th - anchor_vec))
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
           + 2 * wM * M_theta * (th - anchor_vec)
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

    const double _iter_spn  = distance(theta);
    const double _iter_dist = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaTheta, g)
              << "\tSPN energy: " << _iter_spn
              << "\tDistance: " << _iter_dist
              << "\tStep size: " << s;
    const double _iter_self_reg = wM * (theta - anchor_vec).dot(M_theta * (theta - anchor_vec)) + wL * theta.dot(L * theta);
    const double _iter_penalty  = penaltyFunc.eval(theta);
    iter_logger(i, x, _iter_spn, _iter_dist, _iter_self_reg, _iter_penalty);

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
  final_self_reg   = wM * (theta - anchor_vec).dot(M_theta * (theta - anchor_vec)) + wL * theta.dot(L * theta);

  Eigen::MatrixXd V(targetV.rows(), 3);
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      V(i, j) = x(3 * i + j);

  theta1.fromVector(theta);
  return V;
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
    double min_angle_deg,
    const M_Surface_LK& E_surface,
    double nu,
    double h,
    double w_s,
    double w_b,
    const std::vector<int>& ref_faces,
    double& final_distance,
    double& final_spn_energy,
    double& final_self_reg,
    const std::function<void(int, const Eigen::VectorXd&, double, double, double, double)>& iter_logger,
    const std::function<void(const Eigen::VectorXd&)>& callback)
{
  geometry.requireFaceAreas();
  geometry.requireVertexIndices();

  SurfaceMesh& mesh = geometry.mesh;
  const size_t nV = mesh.nVertices();
  const Eigen::Index P_size = static_cast<Eigen::Index>(2 * nV);

  // Pack target / init into flat vectors
  Eigen::VectorXd xTarget(targetV.size());
  for(int i = 0; i < targetV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      xTarget(3 * i + j) = targetV(i, j);
  Eigen::VectorXd x(initV.size());
  for(int i = 0; i < initV.rows(); ++i)
    for(int j = 0; j < 3; ++j)
      x(3 * i + j) = initV(i, j);

  // Pack P into a flat 2|V| vector.  Layout is (P[0].x, P[0].y, P[1].x, ...).
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

  // Helper: write current P_vec back to a |V|x2 MatrixXd
  auto unpack_P = [&](const Eigen::VectorXd& Pv) {
    Eigen::MatrixXd Pm(nV, 2);
    for(int i = 0; i < (int)nV; ++i) {
      Pm(i, 0) = Pv(2 * i + 0);
      Pm(i, 1) = Pv(2 * i + 1);
    }
    return Pm;
  };

  LLTSolver adjointSolver;

  // distance lambda: write P, recompute MrInv, run forward Newton, return SPN.
  // Bails out with +inf if any face has non-positive Mr determinant (P
  // degenerate / inverted); this makes lineSearch reject such steps.
  // Precompute sin(min_angle) threshold once (0 disables the check).
  const double sin_min_angle = (min_angle_deg > 0.0)
                                 ? std::sin(min_angle_deg * M_PI / 180.0)
                                 : 0.0;

  // Helper: smallest sin(angle) over all faces of a packed P (>=0; -1 if any
  // face is degenerate / has a zero-length edge).
  auto min_sin_angle_of = [&](const Eigen::VectorXd& Pv) -> double {
    Eigen::MatrixXd Pm = unpack_P(Pv);
    double s_min = 1.0;
    for(int fi = 0; fi < F.rows(); ++fi) {
      const Eigen::Vector2d p0 = Pm.row(F(fi, 0));
      const Eigen::Vector2d p1 = Pm.row(F(fi, 1));
      const Eigen::Vector2d p2 = Pm.row(F(fi, 2));
      const Eigen::Vector2d e01 = p1 - p0, e02 = p2 - p0, e12 = p2 - p1;
      const double l0 = e01.norm(), l1 = e12.norm(), l2 = e02.norm();
      if (l0 < 1e-12 || l1 < 1e-12 || l2 < 1e-12) return -1.0;
      const double two_area = std::abs(e01.x() * e02.y() - e01.y() * e02.x());
      s_min = std::min({s_min, two_area / (l0 * l2),
                        two_area / (l0 * l1), two_area / (l2 * l1)});
    }
    return s_min;
  };

  // Monotone floor: if the STARTING P is already below the configured
  // min_angle_deg (common for patches whose Parameterize layout has a sliver
  // just under the cap), clamp the guard floor down to the starting quality.
  // Otherwise the very first loop eval distance(P_vec) would return +inf,
  // breaking the Armijo baseline and letting line search accept a garbage
  // step that collapses P (min_angle -> 0, forward sim diverges to ~1e31).
  // With this floor the start is accepted and trial steps may not make the
  // worst triangle any worse than it already is.
  double eff_sin_min_angle = sin_min_angle;
  if (sin_min_angle > 0.0) {
    const double start_sin = min_sin_angle_of(P_vec);
    if (start_sin >= 0.0 && start_sin < eff_sin_min_angle)
      eff_sin_min_angle = start_sin;
  }
  auto distance = [&](const Eigen::VectorXd& Pv) -> double {
    Eigen::MatrixXd Pm = unpack_P(Pv);
    // Per-face quality pre-check: (a) det(Mr) > 0 (no foldover),
    // (b) min angle >= eff_sin_min_angle (monotone floor, see above).
    // Failing either returns +inf so the Armijo line search backtracks.
    for(int fi = 0; fi < F.rows(); ++fi) {
      const Eigen::Vector2d p0 = Pm.row(F(fi, 0));
      const Eigen::Vector2d p1 = Pm.row(F(fi, 1));
      const Eigen::Vector2d p2 = Pm.row(F(fi, 2));
      const Eigen::Vector2d e01 = p1 - p0;
      const Eigen::Vector2d e02 = p2 - p0;
      const double det = e01.x() * e02.y() - e01.y() * e02.x();
      if (!std::isfinite(det) || det < 1e-10)
        return std::numeric_limits<double>::infinity();

      if (eff_sin_min_angle > 0.0) {
        const Eigen::Vector2d e12 = p2 - p1;
        const double l0 = e01.norm(), l1 = e12.norm(), l2 = e02.norm();
        if (l0 < 1e-12 || l1 < 1e-12 || l2 < 1e-12)
          return std::numeric_limits<double>::infinity();
        // 2 * triangle_area = |det|
        const double two_area = std::abs(det);
        // sin(angle at v_i) = 2*area / (l_a * l_b) for the two edges incident to v_i
        const double sin_a0 = two_area / (l0 * l2);   // angle at p0 (between e01, e02)
        const double sin_a1 = two_area / (l0 * l1);   // angle at p1 (between e10, e12)
        const double sin_a2 = two_area / (l2 * l1);   // angle at p2 (between e20, e21)
        const double sin_min = std::min({sin_a0, sin_a1, sin_a2});
        if (sin_min < eff_sin_min_angle)
          return std::numeric_limits<double>::infinity();
      }
    }
    P_io = Pm;
    FaceData<Eigen::Matrix2d> MrInv_curr = precomputeMrInv(
        *dynamic_cast<ManifoldSurfaceMesh*>(&mesh), Pm, F);
    auto simFunc = simulationFunction(geometry, MrInv_curr, lambda_pf, kappa_pf,
                                      E_surface, nu, h, w_s, w_b, ref_faces);
    newton(x, simFunc, adjointSolver, 100, lim, false, fixedIdx);

    const Eigen::VectorXd Pd = Pv - P_anchor_vec;
    const double dist_term      = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    if (!std::isfinite(dist_term))
      return std::numeric_limits<double>::infinity();
    const double anchor_reg     = wM_P * Pd.dot(M_P * Pd);
    const double smoothness_reg = wL_P * Pv.dot(L_P * Pv);
    return dist_term + anchor_reg + smoothness_reg + other_reg;
  };

  // Build x-DOF projection matrix
  Eigen::SparseMatrix<double> P_proj = projectionMatrix(fixedIdx, x.size());

  // theta-block of the KKT system: M_P_kkt = 2 wM_P M_P + 2 wL_P L_P
  Eigen::SparseMatrix<double> M_theta_kkt = 2.0 * wM_P * M_P + 2.0 * wL_P * L_P;

  // Initial joint Hessian (size 3|V| + 2|V|)
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

  // The monotone floor eff_sin_min_angle guarantees the starting P is at or
  // above the guard floor, so this first eval (and the loop baseline f) is
  // always finite -- the Armijo line search has a valid reference.
  double energy0 = distance(P_vec);
  std::cout << "Initial SPN energy (OptP): " << energy0
            << "\t distance: " << (x - xTarget).dot(masses.cwiseProduct(x - xTarget)) << std::endl;
  if (!std::isfinite(energy0))
    std::cerr << "[OptP] initial P energy is non-finite (foldover det<=0); "
                 "OptP cannot start from this layout.\n";

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

    // -----------------------------------------------------------------
    // Two SPN definitions (intentionally distinguished):
    //   SPN2 (OptP internal): dist + wM_P·||Pd||² + wL_P·||P||²_L + other_reg
    //                         -- what OptP actually minimises; required for
    //                            line search and Newton correctness.
    //   SPN1 (logged):        dist + other_reg
    //                         -- excludes the OptP-specific P-anchor /
    //                            P-smoothness regs so the value is on the
    //                            same basis as OptKap / OptLam (which never
    //                            see these terms).  This matches the
    //                            end-of-substage summary written by
    //                            inverse/main.cpp (spn_energy = distance
    //                            + kappa_reg + lambda_reg, line ~772), and
    //                            is what shows up on the convergence curve.
    // -----------------------------------------------------------------
    const double _iter_spn_optp_total = distance(P_vec);        // SPN2
    const double _iter_dist           = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    const Eigen::VectorXd Pd_iter     = P_vec - P_anchor_vec;
    const double _iter_self_reg       = wM_P * Pd_iter.dot(M_P * Pd_iter)
                                      + wL_P * P_vec.dot(L_P * P_vec);
    const double _iter_spn            = _iter_spn_optp_total - _iter_self_reg;  // SPN1

    std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaP, g)
              << "\tSPN energy: " << _iter_spn
              << "\tDistance: " << _iter_dist
              << "\tStep size: " << s;
    iter_logger(i, x, _iter_spn, _iter_dist, _iter_self_reg, 0.0);

    if(TinyAD::newton_decrement(deltaP, g) < lim || solver.info() != Eigen::Success)
      break;

    callback(x);
  }

  // Final unpack: write P_vec back, recompute MrInv, final forward Newton to
  // sync x to equilibrium at the converged P.
  P_io = unpack_P(P_vec);
  FaceData<Eigen::Matrix2d> MrInv_final = precomputeMrInv(
      *dynamic_cast<ManifoldSurfaceMesh*>(&mesh), P_io, F);
  auto simFunc_final = simulationFunction(geometry, MrInv_final, lambda_pf, kappa_pf,
                                          E_surface, nu, h, w_s, w_b, ref_faces);
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
