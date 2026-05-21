#include "newton.h"
#include "functions.h"
#include "parameterization.h"
#include "simulation_utils.h"
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
                                    double E,
                                    double nu,
                                    double h,
                                    double w_s,
                                    double w_b,
                                    const std::vector<int>& ref_faces,
                                    double& final_distance,
                                    double& final_spn_energy,
                                    double& final_self_reg,
                                    const std::function<void(int, double, double)>& iter_logger,
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

    const double _iter_spn  = distance(theta);
    const double _iter_dist = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaTheta, g)
              << "\tSPN energy: " << _iter_spn
              << "\tDistance: " << _iter_dist
              << "\tStep size: " << s << std::endl;
    iter_logger(i, _iter_spn, _iter_dist);
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
                                  const std::function<void(int, double, double)>& iter_logger,
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

    const double _iter_spn  = distance(theta);
    const double _iter_dist = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaTheta, g)
              << "\tSPN energy: " << _iter_spn
              << "\tDistance: " << _iter_dist
              << "\tStep size: " << s << std::endl;
    iter_logger(i, _iter_spn, _iter_dist);

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
                                    const std::function<void(int, double, double)>& iter_logger,
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

    const double _iter_spn  = distance(theta);
    const double _iter_dist = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaTheta, g)
              << "\tSPN energy: " << _iter_spn
              << "\tDistance: " << _iter_dist
              << "\tStep size: " << s << std::endl;
    iter_logger(i, _iter_spn, _iter_dist);
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
                                                          const std::function<void(int, double, double)>& iter_logger,
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

    const double _iter_spn  = distance(theta);
    const double _iter_dist = (x - xTarget).dot(masses.cwiseProduct(x - xTarget));
    std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaTheta, g)
              << "\tSPN energy: " << _iter_spn
              << "\tDistance: " << _iter_dist
              << "\tStep size: " << s << std::endl;
    iter_logger(i, _iter_spn, _iter_dist);

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
