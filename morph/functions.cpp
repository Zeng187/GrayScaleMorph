#include "functions.h"

#include "simulation_utils.h"
#include "material.hpp"
#include "boundary_utils.h"

#include <TinyAD/Utils/Helpers.hh>

#include <vector>

using namespace geometrycentral::surface;

template <typename T, int R, int C>
inline T frob2(const Eigen::Matrix<T, R, C> &M)
{
  return (M.array() * M.array()).sum();
}

// Placeholder functions for theta to lambda/kappa conversion
// TODO: Replace with actual material-based implementation
template <typename T>
inline T theta_to_lambda(T theta)
{
  // Placeholder: assumes theta directly represents stretch ratio
  return theta;
}

template <typename T>
inline T theta_to_kappa(T theta)
{
  // Placeholder: assumes no curvature for now
  return T(0);
}

// ---------------------------------------------------------------------------
// Internal helpers for shape operator (L) computation via ref_faces mapping.
//
// For interior face f, ref_faces[f] == f -> no redirect, identical to inline.
// For boundary face f, ref_faces[f] points to a nearby interior face (BFS in
// buildRefFaces).  The dihedral stencil is taken from that interior face,
// giving a complete (non-truncated) L matrix at boundary.
// ---------------------------------------------------------------------------
namespace {

template <typename T, typename ElementT>
Eigen::Matrix3<T> computeShapeOperator_Sim(
    SurfaceMesh &mesh,
    ElementT &element,
    Face f,
    const std::vector<int> &ref_faces)
{
  Face rf = mesh.face(ref_faces[f.getIndex()]);

  Eigen::Vector3<T> x0_rf = element.variables(rf.halfedge().vertex());
  Eigen::Vector3<T> x1_rf = element.variables(rf.halfedge().next().vertex());
  Eigen::Vector3<T> x2_rf = element.variables(rf.halfedge().next().next().vertex());
  Eigen::Vector3<T> n_rf = (x1_rf - x0_rf).cross(x2_rf - x0_rf);

  Eigen::Matrix3<T> L = Eigen::Matrix3<T>::Zero();
  for (Halfedge he : rf.adjacentHalfedges())
  {
    if (he.edge().isBoundary())
      continue;

    Eigen::Vector3<T> e =
        element.variables(he.next().vertex()) - element.variables(he.vertex());
    Eigen::Vector3<T> nf =
        (element.variables(he.twin().next().next().vertex()) -
         element.variables(he.vertex()))
            .cross(e);

    T theta = atan2(n_rf.cross(nf).dot(e), e.norm() * nf.dot(n_rf));
    Eigen::Vector3<T> t = n_rf.cross(e);
    L += theta * t.normalized() * t.transpose();
  }
  L /= n_rf.squaredNorm();
  return L;
}

template <typename T, typename ElementT>
Eigen::Vector3<T> adjointVertexPos(
    IntrinsicGeometryInterface &geometry,
    ElementT &element,
    Vertex v)
{
  const Eigen::Index vi = geometry.vertexIndices[v];
  Eigen::Vector3<T> pos;
  pos << element.variables(3 * vi),
         element.variables(3 * vi + 1),
         element.variables(3 * vi + 2);
  return pos;
}

template <typename T, typename ElementT>
Eigen::Matrix3<T> computeShapeOperator_Adj(
    IntrinsicGeometryInterface &geometry,
    ElementT &element,
    Face f,
    const std::vector<int> &ref_faces)
{
  SurfaceMesh &mesh = geometry.mesh;
  Face rf = mesh.face(ref_faces[f.getIndex()]);

  Eigen::Vector3<T> x0_rf = adjointVertexPos<T>(geometry, element, rf.halfedge().vertex());
  Eigen::Vector3<T> x1_rf = adjointVertexPos<T>(geometry, element, rf.halfedge().next().vertex());
  Eigen::Vector3<T> x2_rf = adjointVertexPos<T>(geometry, element, rf.halfedge().next().next().vertex());
  Eigen::Vector3<T> n_rf = (x1_rf - x0_rf).cross(x2_rf - x0_rf);

  Eigen::Matrix3<T> L = Eigen::Matrix3<T>::Zero();
  for (Halfedge he : rf.adjacentHalfedges())
  {
    if (he.edge().isBoundary())
      continue;

    Eigen::Vector3<T> e =
        adjointVertexPos<T>(geometry, element, he.next().vertex()) -
        adjointVertexPos<T>(geometry, element, he.vertex());
    Eigen::Vector3<T> nf =
        adjointVertexPos<T>(geometry, element, he.twin().next().next().vertex()) -
        adjointVertexPos<T>(geometry, element, he.vertex());
    nf = nf.cross(e);

    T theta = atan2(n_rf.cross(nf).dot(e), e.norm() * nf.dot(n_rf));
    Eigen::Vector3<T> t = n_rf.cross(e);
    L += theta * t.normalized() * t.transpose();
  }
  L /= n_rf.squaredNorm();
  return L;
}

} // anonymous namespace

TinyAD::ScalarFunction<3, double, Vertex> simulationFunction(IntrinsicGeometryInterface &geometry,
                                                             const FaceData<Eigen::Matrix2d> &MrInv,
                                                             const FaceData<double> &lambda,
                                                             const FaceData<double> &kappa,
                                                             double E, double nu, double h,
                                                             double w_s, double w_b,
                                                             const std::vector<int> &ref_faces)
{

  SurfaceMesh &mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Set up function with 3D vertex positions as variables.
  TinyAD::ScalarFunction<3, double, Vertex> func = TinyAD::scalar_function<3>(mesh.vertices());

  // 1st fundamental form
  func.add_elements<3>(mesh.faces(),
                       [&, alpha, beta, E, nu, h, w_s, w_b, lambda, kappa](auto &element) -> TINYAD_SCALAR_TYPE(element)
                       {
                         using T = TINYAD_SCALAR_TYPE(element);
                         Face f = element.handle;
                         Eigen::Vector3<T> x0 = element.variables(f.halfedge().vertex());
                         Eigen::Vector3<T> x1 = element.variables(f.halfedge().next().vertex());
                         Eigen::Vector3<T> x2 = element.variables(f.halfedge().next().next().vertex());

                         Eigen::Matrix<T, 3, 2> M = TinyAD::col_mat(x1 - x0, x2 - x0);
                         double dA = 0.5 / MrInv[f].determinant();
                         Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f]);
                         Eigen::Matrix<T, 2, 2> a = Ff.transpose() * Ff;

                         T lam = T(lambda[f]);
                         T lam_sqr = lam * lam;
                         Eigen::Matrix<T, 2, 2> Egreen = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();
                         T trM = Egreen.trace();
                         T trM2 = (Egreen * Egreen).trace();
                         T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2);
                         Ws = Ws * (1.0 / lam_sqr);
                         return T(w_s) * Ws * dA;
                       });

  func.add_elements<12>(
      mesh.faces(), [&, alpha, beta, E, nu, h, w_s, w_b, lambda, kappa, ref_faces](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);

        Face f = element.handle;
        Eigen::Vector3<T> x0 = element.variables(f.halfedge().vertex());
        Eigen::Vector3<T> x1 = element.variables(f.halfedge().next().vertex());
        Eigen::Vector3<T> x2 = element.variables(f.halfedge().next().next().vertex());

        Eigen::Matrix<T, 3, 2> M = TinyAD::col_mat(x1 - x0, x2 - x0);
        double dA = 0.5 / MrInv[f].determinant();
        Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f]);

        T lam = T(lambda[f]);
        T kap = T(kappa[f]);

        Eigen::Matrix3<T> L = computeShapeOperator_Sim<T>(mesh, element, f, ref_faces);

        T lam_sqr = lam * lam;
        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2d::Identity();
        Eigen::Matrix2<T> eps_b = (Ff.transpose() * L * Ff) - b_bar;
        T trM = eps_b.trace();
        T trM2 = (eps_b * eps_b).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3);
        Wb = Wb * (1.0 / lam_sqr);

        return T(w_b) * Wb * dA;
      });

  return func;
}

TinyAD::ScalarFunction<3, double, Vertex> simulationFunction(IntrinsicGeometryInterface &geometry,
                                                             const FaceData<Eigen::Matrix2d> &MrInv,
                                                             const FaceData<double> &lambda,
                                                             const VertexData<double> &kappa,
                                                             double E, double nu, double h,
                                                             double w_s, double w_b,
                                                             const std::vector<int> &ref_faces)
{

  SurfaceMesh &mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  TinyAD::ScalarFunction<3, double, Vertex> func = TinyAD::scalar_function<3>(mesh.vertices());

  // 1st fundamental form
  func.add_elements<3>(mesh.faces(),
                       [&, alpha, beta, E, nu, h, w_s, w_b, lambda, kappa](auto &element) -> TINYAD_SCALAR_TYPE(element)
                       {
                         using T = TINYAD_SCALAR_TYPE(element);
                         Face f = element.handle;
                         Eigen::Vector3<T> x0 = element.variables(f.halfedge().vertex());
                         Eigen::Vector3<T> x1 = element.variables(f.halfedge().next().vertex());
                         Eigen::Vector3<T> x2 = element.variables(f.halfedge().next().next().vertex());

                         Eigen::Matrix<T, 3, 2> M = TinyAD::col_mat(x1 - x0, x2 - x0);
                         double dA = 0.5 / MrInv[f].determinant();
                         Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f]);
                         Eigen::Matrix<T, 2, 2> a = Ff.transpose() * Ff;

                         T lam = T(lambda[f]);
                         T lam_sqr = lam * lam;
                         Eigen::Matrix<T, 2, 2> Egreen = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();
                         T trM = Egreen.trace();
                         T trM2 = (Egreen * Egreen).trace();
                         T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2);
                         Ws = Ws * (1.0 / lam_sqr);
                         return T(w_s) * Ws * dA;
                       });

  func.add_elements<12>(
      mesh.faces(), [&, alpha, beta, E, nu, h, w_s, w_b, lambda, kappa, ref_faces](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);

        Face f = element.handle;
        Eigen::Vector3<T> x0 = element.variables(f.halfedge().vertex());
        Eigen::Vector3<T> x1 = element.variables(f.halfedge().next().vertex());
        Eigen::Vector3<T> x2 = element.variables(f.halfedge().next().next().vertex());

        Eigen::Matrix<T, 3, 2> M = TinyAD::col_mat(x1 - x0, x2 - x0);
        double dA = 0.5 / MrInv[f].determinant();
        Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f]);

        T lam = T(lambda[f]);
        T kap = 0.0;
        for(Vertex v: f.adjacentVertices())
          kap += kappa[v] / 3;

        Eigen::Matrix3<T> L = computeShapeOperator_Sim<T>(mesh, element, f, ref_faces);

        T lam_sqr = lam * lam;
        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2d::Identity();
        Eigen::Matrix2<T> eps_b = (Ff.transpose() * L * Ff) - b_bar;
        T trM = eps_b.trace();
        T trM2 = (eps_b * eps_b).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3);
        Wb = Wb * (1.0 / lam_sqr);

        return T(w_b) * Wb * dA;
      });

  return func;
}

TinyAD::ScalarFunction<3, double, Vertex> simulationFunction(IntrinsicGeometryInterface &geometry,
                                                             const FaceData<Eigen::Matrix2d> &MrInv,
                                                             const VertexData<double> &lambda,
                                                             const VertexData<double> &kappa,
                                                             double E,
                                                             double nu,
                                                             double h,
                                                             double w_s,
                                                             double w_b)
{

  SurfaceMesh &mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Set up function with 3D vertex positions as variables.
  TinyAD::ScalarFunction<3, double, Vertex> func = TinyAD::scalar_function<3>(mesh.vertices());

  // 1st fundamental form
  func.add_elements<3>(
      mesh.faces(), [&, alpha, beta, E, nu, h, w_s, w_b, lambda, kappa](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        // Evaluate element using either double or TinyAD::Double
        using T = TINYAD_SCALAR_TYPE(element);

        Face f = element.handle;
        Eigen::Vector3<T> x0 = element.variables(f.halfedge().vertex());
        Eigen::Vector3<T> x1 = element.variables(f.halfedge().next().vertex());
        Eigen::Vector3<T> x2 = element.variables(f.halfedge().next().next().vertex());

        Eigen::Matrix<T, 3, 2> M = TinyAD::col_mat(x1 - x0, x2 - x0);
        double dA = 0.5 / MrInv[f].determinant();

        // Compute deformation gradient
        Eigen::Matrix<T, 3, 2> F = M * (MrInv[f]);

        Eigen::Matrix<T, 2, 2> a = F.transpose() * F;

        T lam = 0.0;
        for(Vertex v: f.adjacentVertices())
          lam += lambda[v] / 3;

        T lam_sqr = lam * lam;
        Eigen::Matrix<T, 2, 2> Egreen = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();
        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();
        T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) ;
        Ws = Ws * (1.0 / lam_sqr);

        return T(w_s) * Ws * dA; });

  func.add_elements<6>(
      mesh.faces(), [&, alpha, beta, E, nu, h, w_s, w_b, lambda, kappa](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        // Evaluate element using either double or TinyAD::Double
        using T = TINYAD_SCALAR_TYPE(element);

        // Get variable 2D vertex positions
        Face f = element.handle;
        auto x0_idx = f.halfedge().vertex();
        auto x1_idx = f.halfedge().next().vertex();
        auto x2_idx = f.halfedge().next().next().vertex();
        Eigen::Vector3<T> x0 = element.variables(f.halfedge().vertex());
        Eigen::Vector3<T> x1 = element.variables(f.halfedge().next().vertex());
        Eigen::Vector3<T> x2 = element.variables(f.halfedge().next().next().vertex());

        Eigen::Matrix<T, 3, 2> M = TinyAD::col_mat(x1 - x0, x2 - x0);
        double dA = 0.5 / MrInv[f].determinant();

        T lam = 0.0;
        for(Vertex v: f.adjacentVertices())
          lam += lambda[v] / 3;
        T kap = 0.0;
        for(Vertex v: f.adjacentVertices())
          kap += kappa[v] / 3;

        // Compute deformation gradient
        Eigen::Matrix<T, 3, 2> F = M * (MrInv[f]);

        // Compute normal
        Eigen::Vector3<T> n = M.col(0).cross(M.col(1));

        Eigen::Matrix3<T> L = Eigen::Matrix3<T>::Zero();
        for(Halfedge he: f.adjacentHalfedges())
        {
          if(he.edge().isBoundary())
            continue;
          Eigen::Vector3<T> e = element.variables(he.next().vertex()) - element.variables(he.vertex());

          // compute dihedral angle theta
          Eigen::Vector3<T> nf =
              (element.variables(he.twin().next().next().vertex()) - element.variables(he.vertex())).cross(e);
          T theta = atan2(n.cross(nf).dot(e), e.norm() * nf.dot(n));

          Eigen::Vector3<T> t = n.cross(e);

          // add edge contribution
          L += theta * t.normalized() * t.transpose();
        }
        L /= n.squaredNorm();

        T lam_sqr = lam * lam;
        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2d::Identity();
        Eigen::Matrix2<T> eps_b = (F.transpose() * L * F) - b_bar;
        T trM = eps_b.trace();
        T trM2 = (eps_b * eps_b).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3);
        Wb = Wb * (1.0 / lam_sqr);

        return T(w_b) * Wb * dA; });

  return func;
}

TinyAD::ScalarFunction<3, double, Vertex> simulationFunctionWithMaterial(
    IntrinsicGeometryInterface &geometry,
    const FaceData<Eigen::Matrix2d> &MrInv,
    const VertexData<double> &t_layer_1,
    const VertexData<double> &t_layer_2,
    const M_Poly_Curve &strain_curve,
    const M_Poly_Curve &moduls_curve,
    double E,
    double nu,
    double h,
    double w_s,
    double w_b)
{
  SurfaceMesh &mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Set up function with 3D vertex positions as variables.
  TinyAD::ScalarFunction<3, double, Vertex> func = TinyAD::scalar_function<3>(mesh.vertices());

  // 1st fundamental form (stretching energy)
  func.add_elements<3>(
      mesh.faces(), [&, alpha, beta, h, w_s, w_b](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);

        Face f = element.handle;
        Eigen::Vector3<T> x0 = element.variables(f.halfedge().vertex());
        Eigen::Vector3<T> x1 = element.variables(f.halfedge().next().vertex());
        Eigen::Vector3<T> x2 = element.variables(f.halfedge().next().next().vertex());

        Eigen::Matrix<T, 3, 2> M = TinyAD::col_mat(x1 - x0, x2 - x0);
        double dA = 0.5 / MrInv[f].determinant();

        // Compute deformation gradient
        Eigen::Matrix<T, 3, 2> F = M * (MrInv[f]);

        // First fundamental form a = F^T F
        Eigen::Matrix<T, 2, 2> a = F.transpose() * F;

        // Compute lambda from t_layer_1, t_layer_2 for each vertex, then average

        double t1_f = 0.0;
        double t2_f = 0.0;
        for(Vertex v: f.adjacentVertices()) {
          double t1_v = t_layer_1[v];
          double t2_v = t_layer_2[v];
          t1_f += t1_v;
          t2_f += t2_v;
        }
        t1_f /= f.degree();
        t2_f /= f.degree();
        double lam = compute_lamb_d(strain_curve, t1_f, t2_f);

        // double lam = 0.0;
        // for(Vertex v: f.adjacentVertices()) {
        //   double t1_v = t_layer_1[v];
        //   double t2_v = t_layer_2[v];
        //   double lam_v = compute_lamb_d(strain_curve, t1_v, t2_v);
        //   lam += lam_v / 3.0;
        // }

        T lam_sqr = lam * lam;
        Eigen::Matrix<T, 2, 2> Egreen = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();
        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();
        T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2);
        Ws = Ws * (1.0 / lam_sqr);
        
        return T(w_s) * Ws * dA; });

  // 2nd fundamental form (bending energy)
  func.add_elements<6>(
      mesh.faces(), [&, alpha, beta, h, w_s, w_b](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);

        Face f = element.handle;
        Eigen::Vector3<T> x0 = element.variables(f.halfedge().vertex());
        Eigen::Vector3<T> x1 = element.variables(f.halfedge().next().vertex());
        Eigen::Vector3<T> x2 = element.variables(f.halfedge().next().next().vertex());

        Eigen::Matrix<T, 3, 2> M = TinyAD::col_mat(x1 - x0, x2 - x0);
        double dA = 0.5 / MrInv[f].determinant();

        // // Compute lambda and kappa from t_layer_1, t_layer_2 for each vertex, then average
        double t1_f = 0.0;
        double t2_f = 0.0;
        for(Vertex v: f.adjacentVertices()) {
          double t1_v = t_layer_1[v]; 
          double t2_v = t_layer_2[v];
          t1_f += t1_v;
          t2_f += t2_v;
        }
        t1_f /= 3;
        t2_f /= 3;
        double lam = compute_lamb_d(strain_curve, t1_f, t2_f);
        double kap = compute_curv_d(strain_curve, h, t1_f, t2_f);

        // double lam = 0.0;
        // double kap = 0.0;
        // for(Vertex v: f.adjacentVertices()) {
        //   double t1_v = t_layer_1[v];
        //   double t2_v = t_layer_2[v];
        //   double lam_v = compute_lamb_d(strain_curve, t1_v, t2_v);
        //   double kap_v = compute_curv_d(strain_curve, h, t1_v, t2_v);
        //   lam += lam_v / 3.0;
        //   kap += kap_v / 3.0;
        // }

        T lam_sqr = lam * lam;

        // Compute deformation gradient
        Eigen::Matrix<T, 3, 2> F = M * (MrInv[f]);

        // Compute normal
        Eigen::Vector3<T> n = M.col(0).cross(M.col(1));

        Eigen::Matrix3<T> L = Eigen::Matrix3<T>::Zero();
        for(Halfedge he: f.adjacentHalfedges())
        {
          if(he.edge().isBoundary())
            continue;
          Eigen::Vector3<T> e = element.variables(he.next().vertex()) - element.variables(he.vertex());

          // compute dihedral angle theta
          Eigen::Vector3<T> nf =
              (element.variables(he.twin().next().next().vertex()) - element.variables(he.vertex())).cross(e);
          T theta = atan2(n.cross(nf).dot(e), e.norm() * nf.dot(n));

          Eigen::Vector3<T> t = n.cross(e);

          // add edge contribution
          L += theta * t.normalized() * t.transpose();
        }
        L /= n.squaredNorm();

        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2d::Identity();
        Eigen::Matrix2<T> eps_b = (F.transpose() * L * F) - b_bar;

        T trM = eps_b.trace();
        T trM2 = (eps_b * eps_b).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3);
        Wb = Wb * (1.0 / lam_sqr);
        return T(w_b) * Wb * dA; });

  return func;
}

TinyAD::ScalarFunction<3, double, Vertex> simulationFunctionWithMaterial(
    IntrinsicGeometryInterface &geometry,
    const FaceData<Eigen::Matrix2d> &MrInv,
    const FaceData<double> &t_layer_1,
    const FaceData<double> &t_layer_2,
    const M_Poly_Curve &strain_curve,
    const M_Poly_Curve &moduls_curve,
    double E,
    double nu,
    double h,
    double w_s,
    double w_b)
{
  SurfaceMesh &mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Set up function with 3D vertex positions as variables.
  TinyAD::ScalarFunction<3, double, Vertex> func = TinyAD::scalar_function<3>(mesh.vertices());

  // 1st fundamental form (stretching energy)
  func.add_elements<3>(
      mesh.faces(), [&, alpha, beta, h, w_s, w_b](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);

        Face f = element.handle;
        Eigen::Vector3<T> x0 = element.variables(f.halfedge().vertex());
        Eigen::Vector3<T> x1 = element.variables(f.halfedge().next().vertex());
        Eigen::Vector3<T> x2 = element.variables(f.halfedge().next().next().vertex());

        Eigen::Matrix<T, 3, 2> M = TinyAD::col_mat(x1 - x0, x2 - x0);
        double dA = 0.5 / MrInv[f].determinant();

        // Compute deformation gradient
        Eigen::Matrix<T, 3, 2> F = M * (MrInv[f]);

        // First fundamental form a = F^T F
        Eigen::Matrix<T, 2, 2> a = F.transpose() * F;

        // Compute lambda from t_layer_1, t_layer_2 for each vertex, then average

        double t1_f = t_layer_1[f];
        double t2_f = t_layer_2[f];

        double lam = compute_lamb_d(strain_curve, t1_f, t2_f);

        T lam_sqr = lam * lam;

        // Green strain E = 0.5*(a_bar^{-1} a - I)
        Eigen::Matrix<T, 2, 2> Egreen = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();

        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();
        T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2);
        Ws = Ws * (1.0 / lam_sqr);
        return T(w_s) * Ws * dA; });

  // 2nd fundamental form (bending energy)
  func.add_elements<6>(
      mesh.faces(), [&, alpha, beta, h, w_s, w_b](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);

        Face f = element.handle;
        Eigen::Vector3<T> x0 = element.variables(f.halfedge().vertex());
        Eigen::Vector3<T> x1 = element.variables(f.halfedge().next().vertex());
        Eigen::Vector3<T> x2 = element.variables(f.halfedge().next().next().vertex());

        Eigen::Matrix<T, 3, 2> M = TinyAD::col_mat(x1 - x0, x2 - x0);
        double dA = 0.5 / MrInv[f].determinant();

        // // Compute lambda and kappa from t_layer_1, t_layer_2 for each vertex, then average
        double t1_f = t_layer_1[f];
        double t2_f = t_layer_2[f];
        double lam = compute_lamb_d(strain_curve, t1_f, t2_f);
        double kap = compute_curv_d(strain_curve, h, t1_f, t2_f);

        T lam_sqr = lam * lam;

        // Compute deformation gradient
        Eigen::Matrix<T, 3, 2> F = M * (MrInv[f]);

        // Compute normal
        Eigen::Vector3<T> n = M.col(0).cross(M.col(1));

        Eigen::Matrix3<T> L = Eigen::Matrix3<T>::Zero();
        for(Halfedge he: f.adjacentHalfedges())
        {
          if(he.edge().isBoundary())
            continue;
          Eigen::Vector3<T> e = element.variables(he.next().vertex()) - element.variables(he.vertex());

          // compute dihedral angle theta
          Eigen::Vector3<T> nf =
              (element.variables(he.twin().next().next().vertex()) - element.variables(he.vertex())).cross(e);
          T theta = atan2(n.cross(nf).dot(e), e.norm() * nf.dot(n));

          Eigen::Vector3<T> t = n.cross(e);

          // add edge contribution
          L += theta * t.normalized() * t.transpose();
        }
        L /= n.squaredNorm();

        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2d::Identity();
        Eigen::Matrix2<T> eps_b = (F.transpose() * L * F) - b_bar;

        T trM = eps_b.trace();
        T trM2 = (eps_b * eps_b).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3);
        Wb = Wb * (1.0 / lam_sqr);
        return T(w_b) * Wb * dA; });

  return func;
}

TinyAD::ScalarFunction<1, double, Eigen::Index> adjointFunction_FixLam_OptKap(IntrinsicGeometryInterface &geometry,
                                                                              const Eigen::MatrixXi &F,
                                                                              const FaceData<Eigen::Matrix2d> &MrInv,
                                                                              const VertexData<double> &lambda,
                                                                              double E,
                                                                              double nu,
                                                                              double h,
                                                                              double w_s,
                                                                              double w_b)
{
  SurfaceMesh &mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Set up function with 3D vertex positions as variables.
  TinyAD::ScalarFunction<1, double, Eigen::Index> func =
      TinyAD::scalar_function<1>(TinyAD::range(3 * mesh.nVertices() + mesh.nVertices()));

  // 1st fundamental form
  func.add_elements<9>(TinyAD::range(F.rows()),
                       [&, alpha, beta, E, nu, h, w_s, w_b, lambda](auto &element) -> TINYAD_SCALAR_TYPE(element)
                       {
                         // Evaluate element using either double or TinyAD::Double
                         using T = TINYAD_SCALAR_TYPE(element);
                         Eigen::Index f_idx = element.handle;

                         // Get 3D vertex positions
                         Eigen::Matrix<T, 3, 2> M;
                         M << element.variables(3 * F(f_idx, 1) + 0) - element.variables(3 * F(f_idx, 0) + 0),
                             element.variables(3 * F(f_idx, 2) + 0) - element.variables(3 * F(f_idx, 0) + 0),
                             element.variables(3 * F(f_idx, 1) + 1) - element.variables(3 * F(f_idx, 0) + 1),
                             element.variables(3 * F(f_idx, 2) + 1) - element.variables(3 * F(f_idx, 0) + 1),
                             element.variables(3 * F(f_idx, 1) + 2) - element.variables(3 * F(f_idx, 0) + 2),
                             element.variables(3 * F(f_idx, 2) + 2) - element.variables(3 * F(f_idx, 0) + 2);

                         double dA = 0.5 / MrInv[f_idx].determinant();

                         Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f_idx]);

                         Eigen::Matrix<T, 2, 2> a = Ff.transpose() * Ff;

                         T lam = 0.0;
                         lam = (lambda[F(f_idx, 0)] + lambda[F(f_idx, 1)] + lambda[F(f_idx, 2)]) / 3.0;
                         T lam_sqr = lam * lam;

                         Eigen::Matrix<T, 2, 2> Egreen = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();

                         T trM = Egreen.trace();
                         T trM2 = (Egreen * Egreen).trace();
                         T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2);
                         Ws = Ws * 1.0 / lam_sqr;
                         return T(w_s) * Ws * dA;
                       });

  // 2nd fundamental form
  geometry.requireVertexIndices();

  func.add_elements<3 * 6 + 3>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, E, nu, h, w_s, w_b, lambda](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        // Evaluate element using either double or TinyAD::Double
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;

        // Get 3D vertex positions
        Eigen::Matrix<T, 3, 2> M;
        M << element.variables(3 * F(f_idx, 1) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 2) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 1) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 2) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 1) + 2) - element.variables(3 * F(f_idx, 0) + 2),
            element.variables(3 * F(f_idx, 2) + 2) - element.variables(3 * F(f_idx, 0) + 2);

        double dA = 0.5 / MrInv[f_idx].determinant();

        // Compute deformation gradient
        Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f_idx]);

        // Compute normal
        Eigen::Vector3<T> n = M.col(0).cross(M.col(1));

        Eigen::Matrix3<T> L = Eigen::Matrix3<T>::Zero();
        Face f = mesh.face(f_idx);
        for (Halfedge he : f.adjacentHalfedges())
        {
          if (he.edge().isBoundary())
            continue;

          // rotate edge e around n
          Eigen::Vector3<T> e;
          e << element.variables(3 * geometry.vertexIndices[he.next().vertex()]) -
                   element.variables(3 * geometry.vertexIndices[he.vertex()]),
              element.variables(3 * geometry.vertexIndices[he.next().vertex()] + 1) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 1),
              element.variables(3 * geometry.vertexIndices[he.next().vertex()] + 2) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 2);

          // compute dihedral angle
          Eigen::Vector3<T> nf;
          nf << element.variables(3 * geometry.vertexIndices[he.twin().next().next().vertex()]) -
                    element.variables(3 * geometry.vertexIndices[he.vertex()]),
              element.variables(3 * geometry.vertexIndices[he.twin().next().next().vertex()] + 1) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 1),
              element.variables(3 * geometry.vertexIndices[he.twin().next().next().vertex()] + 2) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 2);
          nf = nf.cross(e);
          T theta = atan2(n.cross(nf).dot(e), e.norm() * nf.dot(n));

          Eigen::Vector3<T> t = n.cross(e);

          // add edge contribution
          L += theta * t.normalized() * t.transpose();
        }
        L /= n.squaredNorm();

        Eigen::Vector3<T> kappa_f;
        kappa_f << element.variables(3 * mesh.nVertices() + F(f_idx, 0)),
            element.variables(3 * mesh.nVertices() + F(f_idx, 1)),
            element.variables(3 * mesh.nVertices() + F(f_idx, 2));
        T kap = (kappa_f(0) + kappa_f(1) + kappa_f(2)) / 3;

        T lam = 0.0;
        for (Vertex v : f.adjacentVertices())
          lam += lambda[v] / 3;
        T lam_sqr = lam * lam;
        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2d ::Identity();

        // 本征度量 a_bar = (lambda^2) I

        Eigen::Matrix2<T> Egreen = (Ff.transpose() * L * Ff) - b_bar;

        //// std::cout << F.transpose() * L * F << '\n';

        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3);
        Wb = Wb * 1.0 / lam_sqr;
        return T(w_b) * Wb * dA;
      });

  return func;
}

TinyAD::ScalarFunction<1, double, Eigen::Index> adjointFunction_FixLam_OptKap(IntrinsicGeometryInterface &geometry,
                                                                              const Eigen::MatrixXi &F,
                                                                              const FaceData<Eigen::Matrix2d> &MrInv,
                                                                              const FaceData<double> &lambda,
                                                                              double E,
                                                                              double nu,
                                                                              double h,
                                                                              double w_s,
                                                                              double w_b,
                                                                              const std::vector<int> &ref_faces)
{
  SurfaceMesh &mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Variables: [x (3|V|), kappa (|F|)] — kappa is per-face.
  TinyAD::ScalarFunction<1, double, Eigen::Index> func =
      TinyAD::scalar_function<1>(TinyAD::range(3 * mesh.nVertices() + mesh.nFaces()));

  // 1st fundamental form
  func.add_elements<9>(TinyAD::range(F.rows()),
                       [&, alpha, beta, E, nu, h, w_s, w_b, lambda](auto &element) -> TINYAD_SCALAR_TYPE(element)
                       {
                         using T = TINYAD_SCALAR_TYPE(element);
                         Eigen::Index f_idx = element.handle;

                         Eigen::Matrix<T, 3, 2> M;
                         M << element.variables(3 * F(f_idx, 1) + 0) - element.variables(3 * F(f_idx, 0) + 0),
                             element.variables(3 * F(f_idx, 2) + 0) - element.variables(3 * F(f_idx, 0) + 0),
                             element.variables(3 * F(f_idx, 1) + 1) - element.variables(3 * F(f_idx, 0) + 1),
                             element.variables(3 * F(f_idx, 2) + 1) - element.variables(3 * F(f_idx, 0) + 1),
                             element.variables(3 * F(f_idx, 1) + 2) - element.variables(3 * F(f_idx, 0) + 2),
                             element.variables(3 * F(f_idx, 2) + 2) - element.variables(3 * F(f_idx, 0) + 2);

                         double dA = 0.5 / MrInv[f_idx].determinant();
                         Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f_idx]);
                         Eigen::Matrix<T, 2, 2> a = Ff.transpose() * Ff;

                         T lam = T(lambda[f_idx]);
                         T lam_sqr = lam * lam;
                         Eigen::Matrix<T, 2, 2> eps_s = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();

                         T trM = eps_s.trace();
                         T trM2 = (eps_s * eps_s).trace();
                         T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2);
                         Ws = Ws * T(1.0) / lam_sqr;
                         return T(w_s) * Ws * dA;
                       });

  geometry.requireVertexIndices();

  func.add_elements<3 * 6 + 1>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, E, nu, h, w_s, w_b, lambda, ref_faces](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;

        Eigen::Matrix<T, 3, 2> M;
        M << element.variables(3 * F(f_idx, 1) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 2) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 1) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 2) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 1) + 2) - element.variables(3 * F(f_idx, 0) + 2),
            element.variables(3 * F(f_idx, 2) + 2) - element.variables(3 * F(f_idx, 0) + 2);

        double dA = 0.5 / MrInv[f_idx].determinant();
        Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f_idx]);

        Face f = mesh.face(f_idx);
        Eigen::Matrix3<T> L = computeShapeOperator_Adj<T>(geometry, element, f, ref_faces);

        // Per-face kappa: single variable at vars[3|V| + f_idx]
        T kap = element.variables(3 * mesh.nVertices() + f_idx)(0, 0);

        T lam = T(lambda[f]);
        T lam_sqr = lam * lam;
        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2d::Identity();
        Eigen::Matrix2<T> eps_b = (Ff.transpose() * L * Ff) - b_bar;

        T trM = eps_b.trace();
        T trM2 = (eps_b * eps_b).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3);
        Wb = Wb * T(1.0) / lam_sqr;
        return T(w_b) * Wb * dA;
      });

  return func;
}

TinyAD::ScalarFunction<1, double, Eigen::Index> adjointFunction_FixKap_OptLam(IntrinsicGeometryInterface &geometry,
                                                                              const Eigen::MatrixXi &F,
                                                                              const FaceData<Eigen::Matrix2d> &MrInv,
                                                                              const VertexData<double> &kappa,
                                                                              double E,
                                                                              double nu,
                                                                              double h,
                                                                              double w_s,
                                                                              double w_b)
{
  SurfaceMesh &mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Set up function with 3D vertex positions as variables.
  TinyAD::ScalarFunction<1, double, Eigen::Index> func =
      TinyAD::scalar_function<1>(TinyAD::range(3 * mesh.nVertices() + mesh.nVertices()));

  // 1st fundamental form
  func.add_elements<6 + 6>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, E, nu, h, w_s, w_b](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        // Evaluate element using either double or TinyAD::Double
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;

        // Get 3D vertex positions
        Eigen::Matrix<T, 3, 2> M;
        M << element.variables(3 * F(f_idx, 1) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 2) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 1) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 2) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 1) + 2) - element.variables(3 * F(f_idx, 0) + 2),
            element.variables(3 * F(f_idx, 2) + 2) - element.variables(3 * F(f_idx, 0) + 2);

        double dA = 0.5 / MrInv[f_idx].determinant();

        // Compute deformation gradient
        Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f_idx]);

        // 第一基本型 a = F^T F
        Eigen::Matrix<T, 2, 2> a = Ff.transpose() * Ff;

        // 本征度量 a_bar = (lambda^2) I
        Eigen::Vector3<T> lambda_f;
        lambda_f << element.variables(3 * mesh.nVertices() + F(f_idx, 0)),
            element.variables(3 * mesh.nVertices() + F(f_idx, 1)),
            element.variables(3 * mesh.nVertices() + F(f_idx, 2));
        T lam = (lambda_f(0) + lambda_f(1) + lambda_f(2)) / 3;
        T lam_sqr = lam * lam;

        Eigen::Matrix<T, 2, 2> Egreen = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();

        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();
        T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2);
        Ws = Ws * 1.0 / lam_sqr;
        return T(w_s) * Ws * dA;
      });

  // 2nd fundamental form
  geometry.requireVertexIndices();

  func.add_elements<3 * 6 + 3>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, E, nu, h, w_s, w_b, kappa](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        // Evaluate element using either double or TinyAD::Double
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;

        // Get 3D vertex positions
        Eigen::Matrix<T, 3, 2> M;
        M << element.variables(3 * F(f_idx, 1) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 2) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 1) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 2) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 1) + 2) - element.variables(3 * F(f_idx, 0) + 2),
            element.variables(3 * F(f_idx, 2) + 2) - element.variables(3 * F(f_idx, 0) + 2);

        double dA = 0.5 / MrInv[f_idx].determinant();

        // Compute deformation gradient
        Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f_idx]);

        // Compute normal
        Eigen::Vector3<T> n = M.col(0).cross(M.col(1));

        Eigen::Matrix3<T> L = Eigen::Matrix3<T>::Zero();
        Face f = mesh.face(f_idx);
        for (Halfedge he : f.adjacentHalfedges())
        {
          if (he.edge().isBoundary())
            continue;

          // rotate edge e around n
          Eigen::Vector3<T> e;
          e << element.variables(3 * geometry.vertexIndices[he.next().vertex()]) -
                   element.variables(3 * geometry.vertexIndices[he.vertex()]),
              element.variables(3 * geometry.vertexIndices[he.next().vertex()] + 1) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 1),
              element.variables(3 * geometry.vertexIndices[he.next().vertex()] + 2) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 2);

          // compute dihedral angle
          Eigen::Vector3<T> nf;
          nf << element.variables(3 * geometry.vertexIndices[he.twin().next().next().vertex()]) -
                    element.variables(3 * geometry.vertexIndices[he.vertex()]),
              element.variables(3 * geometry.vertexIndices[he.twin().next().next().vertex()] + 1) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 1),
              element.variables(3 * geometry.vertexIndices[he.twin().next().next().vertex()] + 2) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 2);
          nf = nf.cross(e);
          T theta = atan2(n.cross(nf).dot(e), e.norm() * nf.dot(n));

          Eigen::Vector3<T> t = n.cross(e);

          // add edge contribution
          L += theta * t.normalized() * t.transpose();
        }
        L /= n.squaredNorm();

        Eigen::Vector3<T> lambda_f;
        lambda_f << element.variables(3 * mesh.nVertices() + F(f_idx, 0)),
            element.variables(3 * mesh.nVertices() + F(f_idx, 1)),
            element.variables(3 * mesh.nVertices() + F(f_idx, 2));
        T lam = (lambda_f(0) + lambda_f(1) + lambda_f(2)) / 3;
        T lam_sqr = lam * lam;
        T kap = 0.0;
        for (Vertex v : f.adjacentVertices())
          kap += kappa[v] / 3;

        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2d ::Identity();

        Eigen::Matrix2<T> Egreen = (Ff.transpose() * L * Ff) - b_bar;

        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3);
        Wb = Wb * 1.0 / lam_sqr;
        return T(w_b) * Wb * dA;
      });

  return func;
}

TinyAD::ScalarFunction<1, double, Eigen::Index>
adjointFunction_FixKap_OptLam2(IntrinsicGeometryInterface &geometry,
                               const Eigen::MatrixXi &F,
                               const FaceData<Eigen::Matrix2d> &MrInv,
                               const FaceData<double> &kappa, // FixKap: per-face constant
                               double E,
                               double nu,
                               double h,
                               double w_s,
                               double w_b,
                               const std::vector<int> &ref_faces)
{
  SurfaceMesh &mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Variables: [x (3|V|), lambda (|F|)]
  TinyAD::ScalarFunction<1, double, Eigen::Index> func =
      TinyAD::scalar_function<1>(TinyAD::range(3 * mesh.nVertices() + mesh.nFaces()));

  // -----------------------------
  // 1st fundamental form (stretching)
  // -----------------------------
  func.add_elements<6 + 6>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, E, nu, h, w_s, w_b](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;

        // 3D vertex positions on this face
        Eigen::Matrix<T, 3, 2> M;
        M << element.variables(3 * F(f_idx, 1) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 2) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 1) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 2) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 1) + 2) - element.variables(3 * F(f_idx, 0) + 2),
            element.variables(3 * F(f_idx, 2) + 2) - element.variables(3 * F(f_idx, 0) + 2);

        double dA = 0.5 / MrInv[f_idx].determinant();

        // deformation gradient
        Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f_idx]);

        // a = F^T F
        Eigen::Matrix<T, 2, 2> a = Ff.transpose() * Ff;

        // -------- Face-based lambda --------
        // variable layout: lambda_f = vars[3|V| + f_idx]
        T lam = element.variables(3 * mesh.nVertices() + f_idx)(0, 0);
        T lam_sqr = lam * lam;

        // a_bar = (lambda^2) I
        Eigen::Matrix<T, 2, 2> Egreen = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();

        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();

        // same as your original code
        T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2);
        Ws = Ws * T(1.0) / lam_sqr;

        return T(w_s) * Ws * dA;
      });

  // -----------------------------
  // 2nd fundamental form (bending)
  // -----------------------------
  geometry.requireVertexIndices();

  func.add_elements<3 * 6 + 3>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, E, nu, h, w_s, w_b, kappa, ref_faces](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;

        Eigen::Matrix<T, 3, 2> M;
        M << element.variables(3 * F(f_idx, 1) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 2) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 1) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 2) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 1) + 2) - element.variables(3 * F(f_idx, 0) + 2),
            element.variables(3 * F(f_idx, 2) + 2) - element.variables(3 * F(f_idx, 0) + 2);

        double dA = 0.5 / MrInv[f_idx].determinant();
        Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f_idx]);

        Face f = mesh.face(f_idx);
        Eigen::Matrix3<T> L = computeShapeOperator_Adj<T>(geometry, element, f, ref_faces);

        T lam = element.variables(3 * mesh.nVertices() + f_idx)(0, 0);
        T lam_sqr = lam * lam;

        T kap = T(kappa[f]);

        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2d::Identity();
        Eigen::Matrix2<T> Egreen = (Ff.transpose() * L * Ff) - b_bar;

        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();

        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3);
        Wb = Wb * T(1.0) / lam_sqr;

        return T(w_b) * Wb * dA;
      });

  return func;
}

TinyAD::ScalarFunction<1, double, Eigen::Index> adjointFunctionWithMaterial_Lay1(
    IntrinsicGeometryInterface &geometry,
    const Eigen::MatrixXi &F,
    const FaceData<Eigen::Matrix2d> &MrInv,
    const VertexData<double> &t_layer_1,
    const M_Poly_Curve &strain_curve,
    const M_Poly_Curve &moduls_curve,
    double E,
    double nu,
    double h,
    double w_s,
    double w_b)
{
  SurfaceMesh &mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Set up function with 3D vertex positions + t_layer_2 as variables
  // Variables layout: [x0,y0,z0, x1,y1,z1, ..., t2_0, t2_1, ...]
  TinyAD::ScalarFunction<1, double, Eigen::Index> func =
      TinyAD::scalar_function<1>(TinyAD::range(3 * mesh.nVertices() + mesh.nVertices()));

  // 1st fundamental form (stretching energy)
  func.add_elements<9 + 3>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, h, w_s, w_b](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;

        // Get 3D vertex positions
        Eigen::Matrix<T, 3, 2> M;
        M << element.variables(3 * F(f_idx, 1) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 2) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 1) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 2) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 1) + 2) - element.variables(3 * F(f_idx, 0) + 2),
            element.variables(3 * F(f_idx, 2) + 2) - element.variables(3 * F(f_idx, 0) + 2);

        double dA = 0.5 / MrInv[f_idx].determinant();

        Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f_idx]);
        Eigen::Matrix<T, 2, 2> a = Ff.transpose() * Ff;

        // Get t_layer_1 (fixed) and t_layer_2 (variable) for face vertices
        T t1_f = T(0);
        T t2_f = T(0);
        for (int k = 0; k < 3; k++)
        {
          double t1_v = t_layer_1[mesh.vertex(F(f_idx, k))];
          T t2_v = element.variables(3 * mesh.nVertices() + F(f_idx, k))(0);
          t1_f += T(t1_v);
          t2_f += t2_v;
        }
        t1_f /= T(3);
        t2_f /= T(3);
        T lam = compute_lamb_d(strain_curve, t1_f, t2_f);

        // T lam = T(0);
        // T kap = T(0);
        // for (int k = 0; k < 3; k++) {
        //   T t1_v = t_layer_1[mesh.vertex(F(f_idx, k))];
        //   T t2_v = element.variables(3 * mesh.nVertices() + F(f_idx, k))(0);
        //   T lam_v = compute_lamb_d(strain_curve, t1_v, t2_v);
        //   T kap_v = compute_curv_d(strain_curve, h, t1_v, t2_v);
        //   lam += lam_v / T(3.0);
        //   kap += kap_v / T(3.0);
        // }

        T lam_sqr = lam * lam;

        // Green strain
        Eigen::Matrix<T, 2, 2> Egreen = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();

        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();
        T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2);
        Ws = Ws * T(1.0) / lam_sqr;
        return T(w_s) * Ws * dA;
      });

  // 2nd fundamental form (bending energy)
  geometry.requireVertexIndices();

  func.add_elements<3 * 6 + 3>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, h, w_s, w_b](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;

        // Get 3D vertex positions
        Eigen::Matrix<T, 3, 2> M;
        M << element.variables(3 * F(f_idx, 1) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 2) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 1) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 2) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 1) + 2) - element.variables(3 * F(f_idx, 0) + 2),
            element.variables(3 * F(f_idx, 2) + 2) - element.variables(3 * F(f_idx, 0) + 2);

        double dA = 0.5 / MrInv[f_idx].determinant();

        // Compute deformation gradient
        Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f_idx]);

        // Compute normal
        Eigen::Vector3<T> n = M.col(0).cross(M.col(1));

        Eigen::Matrix3<T> L = Eigen::Matrix3<T>::Zero();
        Face f = mesh.face(f_idx);
        for (Halfedge he : f.adjacentHalfedges())
        {
          if (he.edge().isBoundary())
            continue;

          Eigen::Vector3<T> e;
          e << element.variables(3 * geometry.vertexIndices[he.next().vertex()]) -
                   element.variables(3 * geometry.vertexIndices[he.vertex()]),
              element.variables(3 * geometry.vertexIndices[he.next().vertex()] + 1) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 1),
              element.variables(3 * geometry.vertexIndices[he.next().vertex()] + 2) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 2);

          Eigen::Vector3<T> nf;
          nf << element.variables(3 * geometry.vertexIndices[he.twin().next().next().vertex()]) -
                    element.variables(3 * geometry.vertexIndices[he.vertex()]),
              element.variables(3 * geometry.vertexIndices[he.twin().next().next().vertex()] + 1) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 1),
              element.variables(3 * geometry.vertexIndices[he.twin().next().next().vertex()] + 2) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 2);
          nf = nf.cross(e);
          T theta = atan2(n.cross(nf).dot(e), e.norm() * nf.dot(n));

          Eigen::Vector3<T> t = n.cross(e);
          L += theta * t.normalized() * t.transpose();
        }
        L /= n.squaredNorm();

        // // Get t_layer_1 (fixed) and t_layer_2 (variable) for face vertices
        T t1_f = T(0);
        T t2_f = T(0);
        for (int k = 0; k < 3; k++)
        {
          double t1_v = t_layer_1[mesh.vertex(F(f_idx, k))];
          T t2_v = element.variables(3 * mesh.nVertices() + F(f_idx, k))(0);
          t1_f += T(t1_v);
          t2_f += t2_v;
        }
        t1_f /= T(3);
        t2_f /= T(3);
        T lam = compute_lamb_d(strain_curve, t1_f, t2_f);
        T kap = compute_curv_d(strain_curve, h, t1_f, t2_f);

        // T lam = T(0);
        // T kap = T(0);
        // for (int k = 0; k < 3; k++) {
        //   T t1_v = t_layer_1[mesh.vertex(F(f_idx, k))];
        //   T t2_v = element.variables(3 * mesh.nVertices() + F(f_idx, k))(0);
        //   T lam_v = compute_lamb_d(strain_curve, t1_v, t2_v);
        //   T kap_v = compute_curv_d(strain_curve, h, t1_v, t2_v);
        //   lam += lam_v / T(3.0);
        //   kap += kap_v / T(3.0);
        // }

        T lam_sqr = lam * lam;
        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2<T>::Identity();

        Eigen::Matrix2<T> eps_b = (Ff.transpose() * L * Ff) - b_bar;

        T trM = eps_b.trace();
        T trM2 = (eps_b * eps_b).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3);
        Wb = Wb * T(1.0) / lam_sqr;
        return T(w_b) * Wb * dA;
      });

  return func;
}

TinyAD::ScalarFunction<1, double, Eigen::Index> adjointFunctionWithMaterial_Lay2(
    IntrinsicGeometryInterface &geometry,
    const Eigen::MatrixXi &F,
    const FaceData<Eigen::Matrix2d> &MrInv,
    const VertexData<double> &t_layer_2,
    const M_Poly_Curve &strain_curve,
    const M_Poly_Curve &moduls_curve,
    double E,
    double nu,
    double h,
    double w_s,
    double w_b)
{
  SurfaceMesh &mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Set up function with 3D vertex positions + t_layer_2 as variables
  // Variables layout: [x0,y0,z0, x1,y1,z1, ..., t2_0, t2_1, ...]
  TinyAD::ScalarFunction<1, double, Eigen::Index> func =
      TinyAD::scalar_function<1>(TinyAD::range(3 * mesh.nVertices() + mesh.nVertices()));

  // 1st fundamental form (stretching energy)
  func.add_elements<9 + 3>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, h, w_s, w_b](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;

        // Get 3D vertex positions
        Eigen::Matrix<T, 3, 2> M;
        M << element.variables(3 * F(f_idx, 1) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 2) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 1) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 2) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 1) + 2) - element.variables(3 * F(f_idx, 0) + 2),
            element.variables(3 * F(f_idx, 2) + 2) - element.variables(3 * F(f_idx, 0) + 2);

        double dA = 0.5 / MrInv[f_idx].determinant();

        Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f_idx]);
        Eigen::Matrix<T, 2, 2> a = Ff.transpose() * Ff;

        // Get t_layer_1 (fixed) and t_layer_2 (variable) for face vertices
        T t1_f = T(0);
        T t2_f = T(0);
        for (int k = 0; k < 3; k++)
        {
          T t1_v = element.variables(3 * mesh.nVertices() + F(f_idx, k))(0);
          double t2_v = t_layer_2[mesh.vertex(F(f_idx, k))];
          t1_f += T(t1_v);
          t2_f += t2_v;
        }
        t1_f /= T(3);
        t2_f /= T(3);
        T lam = compute_lamb_d(strain_curve, t1_f, t2_f);

        // T lam = T(0);
        // T kap = T(0);
        // for (int k = 0; k < 3; k++) {
        //   T t1_v = t_layer_1[mesh.vertex(F(f_idx, k))];
        //   T t2_v = element.variables(3 * mesh.nVertices() + F(f_idx, k))(0);
        //   T lam_v = compute_lamb_d(strain_curve, t1_v, t2_v);
        //   T kap_v = compute_curv_d(strain_curve, h, t1_v, t2_v);
        //   lam += lam_v / T(3.0);
        //   kap += kap_v / T(3.0);
        // }

        T lam_sqr = lam * lam;

        // Green strain
        Eigen::Matrix<T, 2, 2> Egreen = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();

        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();
        T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2);
        Ws = Ws * T(1.0) / lam_sqr;
        return T(w_s) * Ws * dA;
      });

  // 2nd fundamental form (bending energy)
  geometry.requireVertexIndices();

  func.add_elements<3 * 6 + 3>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, h, w_s, w_b](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;

        // Get 3D vertex positions
        Eigen::Matrix<T, 3, 2> M;
        M << element.variables(3 * F(f_idx, 1) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 2) + 0) - element.variables(3 * F(f_idx, 0) + 0),
            element.variables(3 * F(f_idx, 1) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 2) + 1) - element.variables(3 * F(f_idx, 0) + 1),
            element.variables(3 * F(f_idx, 1) + 2) - element.variables(3 * F(f_idx, 0) + 2),
            element.variables(3 * F(f_idx, 2) + 2) - element.variables(3 * F(f_idx, 0) + 2);

        double dA = 0.5 / MrInv[f_idx].determinant();

        // Compute deformation gradient
        Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f_idx]);

        // Compute normal
        Eigen::Vector3<T> n = M.col(0).cross(M.col(1));

        Eigen::Matrix3<T> L = Eigen::Matrix3<T>::Zero();
        Face f = mesh.face(f_idx);
        for (Halfedge he : f.adjacentHalfedges())
        {
          if (he.edge().isBoundary())
            continue;

          Eigen::Vector3<T> e;
          e << element.variables(3 * geometry.vertexIndices[he.next().vertex()]) -
                   element.variables(3 * geometry.vertexIndices[he.vertex()]),
              element.variables(3 * geometry.vertexIndices[he.next().vertex()] + 1) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 1),
              element.variables(3 * geometry.vertexIndices[he.next().vertex()] + 2) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 2);

          Eigen::Vector3<T> nf;
          nf << element.variables(3 * geometry.vertexIndices[he.twin().next().next().vertex()]) -
                    element.variables(3 * geometry.vertexIndices[he.vertex()]),
              element.variables(3 * geometry.vertexIndices[he.twin().next().next().vertex()] + 1) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 1),
              element.variables(3 * geometry.vertexIndices[he.twin().next().next().vertex()] + 2) -
                  element.variables(3 * geometry.vertexIndices[he.vertex()] + 2);
          nf = nf.cross(e);
          T theta = atan2(n.cross(nf).dot(e), e.norm() * nf.dot(n));

          Eigen::Vector3<T> t = n.cross(e);
          L += theta * t.normalized() * t.transpose();
        }
        L /= n.squaredNorm();

        // // Get t_layer_1 (fixed) and t_layer_2 (variable) for face vertices
        T t1_f = T(0);
        T t2_f = T(0);
        for (int k = 0; k < 3; k++)
        {
          T t1_v = element.variables(3 * mesh.nVertices() + F(f_idx, k))(0);
          double t2_v = t_layer_2[mesh.vertex(F(f_idx, k))];
          t1_f += T(t1_v);
          t2_f += t2_v;
        }
        t1_f /= T(3);
        t2_f /= T(3);
        T lam = compute_lamb_d(strain_curve, t1_f, t2_f);
        T kap = compute_curv_d(strain_curve, h, t1_f, t2_f);

        // T lam = T(0);
        // T kap = T(0);
        // for (int k = 0; k < 3; k++) {
        //   T t1_v = t_layer_1[mesh.vertex(F(f_idx, k))];
        //   T t2_v = element.variables(3 * mesh.nVertices() + F(f_idx, k))(0);
        //   T lam_v = compute_lamb_d(strain_curve, t1_v, t2_v);
        //   T kap_v = compute_curv_d(strain_curve, h, t1_v, t2_v);
        //   lam += lam_v / T(3.0);
        //   kap += kap_v / T(3.0);
        // }

        T lam_sqr = lam * lam;
        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2<T>::Identity();

        Eigen::Matrix2<T> eps_b = (Ff.transpose() * L * Ff) - b_bar;

        T trM = eps_b.trace();
        T trM2 = (eps_b * eps_b).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3);
        Wb = Wb * T(1.0) / lam_sqr;
        return T(w_b) * Wb * dA;
      });

  return func;
}

TinyAD::ScalarFunction<1, double, Eigen::Index> MaterialPenaltyFunctionPerV(IntrinsicGeometryInterface &geometry,
                                                                            const std::vector<double> &feasible_vals,
                                                                            double beta)
{
  SurfaceMesh &mesh = geometry.mesh;

  int nV = mesh.nVertices();
  auto func = TinyAD::scalar_function<1>(TinyAD::range(mesh.nVertices()));

  int feasible_cnt = feasible_vals.size();
  func.add_elements<1>(TinyAD::range(mesh.nVertices()),
                       [&, feasible_vals, feasible_cnt, beta, nV](auto &element) -> TINYAD_SCALAR_TYPE(element)
                       {
                         using T = TINYAD_SCALAR_TYPE(element);
                         Eigen::Index v_idx = element.handle;
                         T theta = element.variables(v_idx)(0);

                         T r = T(0.0);
                         // for(int j = 0; j < feasible_cnt; ++j)
                         // {
                         //   T theta_j = T(feasible_vals[j]);
                         //   T diff = theta - theta_j;
                         //   r += exp(-T(beta) * diff * diff);
                         // }
                         r = 1e6;
                         for (int j = 0; j < feasible_cnt; ++j)
                         {
                           T theta_j = T(feasible_vals[j]);
                           T diff = theta - theta_j;
                           T sqdiff = diff * diff;
                           if (sqdiff < r)
                             r = sqdiff;
                         }
                         r = exp(-T(beta) * r);

                         return -log(r + T(1e-12)) / T(nV);
                       });

  return func;
}

TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialPenaltyFunctionPerF(IntrinsicGeometryInterface &geometry,
                            const std::vector<double> &feasible_vals,
                            double beta)
{
  SurfaceMesh &mesh = geometry.mesh;

  const int nF = static_cast<int>(mesh.nFaces());
  auto func = TinyAD::scalar_function<1>(TinyAD::range(mesh.nFaces()));

  const int feasible_cnt = static_cast<int>(feasible_vals.size());

  func.add_elements<1>(
      TinyAD::range(mesh.nFaces()),
      [&, feasible_vals, feasible_cnt, beta, nF](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;
        T theta = element.variables(f_idx)(0);

        // Hard min-distance² penalty (per-face, averaged): the argmin selects
        // the nearest feasible value as a value-only comparison, while TinyAD
        // tracks autodiff only through the surviving (theta - cand)² branch.
        // Result:
        //   gradient = (2·beta/nF) · (theta - nearest_candidate)
        // strictly points toward the projection target, no soft-mixing of
        // neighbouring candidates and no saturation at distance.
        T r = T(1e30);
        for (int j = 0; j < feasible_cnt; ++j)
        {
          T theta_j = T(feasible_vals[j]);
          T diff = theta - theta_j;
          T sqdiff = diff * diff;
          if (sqdiff < r)
            r = sqdiff;
        }
        return T(beta) * r / T(nF);
      });

  return func;
}


TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialPenaltyFunctionPerF_Efrati(IntrinsicGeometryInterface &geometry,
                                   const std::vector<double> &cand_self,
                                   const std::vector<double> &cand_other,
                                   const std::vector<double> &other_const,
                                   double E,
                                   double nu,
                                   double h,
                                   double beta,
                                   bool self_is_kappa)
{
  SurfaceMesh &mesh = geometry.mesh;
  const int nF      = static_cast<int>(mesh.nFaces());
  const int cnt     = static_cast<int>(cand_self.size());

  const double C_pre  = E / (1.0 - nu);
  const double C_str  = h / 4.0;
  const double C_bend = (h * h * h) / 12.0;

  auto func = TinyAD::scalar_function<1>(TinyAD::range(mesh.nFaces()));
  func.add_elements<1>(
      TinyAD::range(mesh.nFaces()),
      [&, cand_self, cand_other, other_const,
       beta, nF, cnt, C_pre, C_str, C_bend, self_is_kappa]
      (auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;
        T theta = element.variables(f_idx)(0);
        const double other_f = other_const[static_cast<size_t>(f_idx)];

        T r = T(1e30);
        for (int j = 0; j < cnt; ++j)
        {
          const double lam_bar = self_is_kappa ? cand_other[j] : cand_self[j];
          const double kap_bar = self_is_kappa ? cand_self[j]  : cand_other[j];
          const double lam_bar_sq = lam_bar * lam_bar;

          T lam, kap;
          if (self_is_kappa) { lam = T(other_f); kap = theta;        }
          else               { lam = theta;       kap = T(other_f);   }

          T lam_str  = lam * lam - T(lam_bar_sq);    // (lambda^2 - lambda_bar^2)
          T kap_bend = kap - T(kap_bar);              // (kappa - kappa_bar)
          T d2 = T(lam_bar_sq) * T(C_pre)
               * (T(C_str)  * lam_str  * lam_str
                + T(C_bend) * kap_bend * kap_bend);
          if (d2 < r) r = d2;
        }
        return T(beta) * r / T(nF);
      });

  return func;
}


TinyAD::ScalarFunction<1, double, Eigen::Index>
MaterialPenaltyFunctionPerF_Joint2D(IntrinsicGeometryInterface &geometry,
                                    const std::vector<double> &cand_self,
                                    const std::vector<double> &cand_other,
                                    const std::vector<double> &other_const,
                                    double beta,
                                    bool self_is_kappa)
{
  SurfaceMesh &mesh = geometry.mesh;
  const int nF  = static_cast<int>(mesh.nFaces());
  const int cnt = static_cast<int>(cand_self.size());
  (void)self_is_kappa;  // symmetric formula: which variable is "self" doesn't matter for the 2D Euclidean form

  auto func = TinyAD::scalar_function<1>(TinyAD::range(mesh.nFaces()));
  func.add_elements<1>(
      TinyAD::range(mesh.nFaces()),
      [&, cand_self, cand_other, other_const, beta, nF, cnt]
      (auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;
        T theta = element.variables(f_idx)(0);
        const double other_f = other_const[static_cast<size_t>(f_idx)];

        T r = T(1e30);
        for (int j = 0; j < cnt; ++j)
        {
          const double s_bar = cand_self[j];
          const double o_bar = cand_other[j];
          T      d_self  = theta - T(s_bar);
          double d_other = other_f - o_bar;
          T d2 = d_self * d_self + T(d_other * d_other);
          if (d2 < r) r = d2;
        }
        return T(beta) * r / T(nF);
      });

  return func;
}


// ---------------------------------------------------------------------------
// adjointFunction_FixMaterial_OptP
// ---------------------------------------------------------------------------
// Variables layout (TinyAD 1D scalar array of size 3|V| + 2|V|):
//   indices [0, 3|V|)        -> x  (per-vertex 3D position)
//   indices [3|V|, 3|V|+2|V|) -> P  (per-vertex 2D parameterisation)
// Constants per face: lambda_pf[f], kappa_pf[f].
// P enters through Mr = [P1-P0, P2-P0] per face, MrInv = Mr^{-1}, dA = 0.5
// * det(Mr).  TinyAD autodiffs through the 2x2 inverse and determinant.
TinyAD::ScalarFunction<1, double, Eigen::Index>
adjointFunction_FixMaterial_OptP(IntrinsicGeometryInterface &geometry,
                                 const Eigen::MatrixXi &F,
                                 const FaceData<double> &lambda_pf,
                                 const FaceData<double> &kappa_pf,
                                 double E,
                                 double nu,
                                 double h,
                                 double w_s,
                                 double w_b,
                                 const std::vector<int> &ref_faces)
{
  SurfaceMesh &mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta  = E / (2 * (1 + nu));
  const int    nV    = static_cast<int>(mesh.nVertices());
  const int    pOff  = 3 * nV;   // P variables start at this offset

  TinyAD::ScalarFunction<1, double, Eigen::Index> func =
      TinyAD::scalar_function<1>(TinyAD::range(3 * nV + 2 * nV));

  // -- Stretching term --
  func.add_elements<15>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, w_s, lambda_pf, pOff](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;

        const int v0 = F(f_idx, 0);
        const int v1 = F(f_idx, 1);
        const int v2 = F(f_idx, 2);

        Eigen::Matrix<T, 3, 2> M;
        M << element.variables(3 * v1 + 0) - element.variables(3 * v0 + 0),
             element.variables(3 * v2 + 0) - element.variables(3 * v0 + 0),
             element.variables(3 * v1 + 1) - element.variables(3 * v0 + 1),
             element.variables(3 * v2 + 1) - element.variables(3 * v0 + 1),
             element.variables(3 * v1 + 2) - element.variables(3 * v0 + 2),
             element.variables(3 * v2 + 2) - element.variables(3 * v0 + 2);

        Eigen::Matrix<T, 2, 2> Mr;
        Mr << element.variables(pOff + 2 * v1 + 0) - element.variables(pOff + 2 * v0 + 0),
              element.variables(pOff + 2 * v2 + 0) - element.variables(pOff + 2 * v0 + 0),
              element.variables(pOff + 2 * v1 + 1) - element.variables(pOff + 2 * v0 + 1),
              element.variables(pOff + 2 * v2 + 1) - element.variables(pOff + 2 * v0 + 1);

        Eigen::Matrix<T, 2, 2> MrInv = Mr.inverse();
        T dA = T(0.5) * Mr.determinant();

        Eigen::Matrix<T, 3, 2> Ff = M * MrInv;
        Eigen::Matrix<T, 2, 2> a  = Ff.transpose() * Ff;

        T lam     = T(lambda_pf[f_idx]);
        T lam_sqr = lam * lam;
        Eigen::Matrix<T, 2, 2> eps_s = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();
        T trM  = eps_s.trace();
        T trM2 = (eps_s * eps_s).trace();
        T Ws   = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) / lam_sqr;
        return T(w_s) * Ws * dA;
      });

  // -- Bending term --
  geometry.requireVertexIndices();
  func.add_elements<3 * 6 + 3 * 2>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, h, w_b, lambda_pf, kappa_pf, ref_faces, pOff](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Index f_idx = element.handle;

        const int v0 = F(f_idx, 0);
        const int v1 = F(f_idx, 1);
        const int v2 = F(f_idx, 2);

        Eigen::Matrix<T, 3, 2> M;
        M << element.variables(3 * v1 + 0) - element.variables(3 * v0 + 0),
             element.variables(3 * v2 + 0) - element.variables(3 * v0 + 0),
             element.variables(3 * v1 + 1) - element.variables(3 * v0 + 1),
             element.variables(3 * v2 + 1) - element.variables(3 * v0 + 1),
             element.variables(3 * v1 + 2) - element.variables(3 * v0 + 2),
             element.variables(3 * v2 + 2) - element.variables(3 * v0 + 2);

        Eigen::Matrix<T, 2, 2> Mr;
        Mr << element.variables(pOff + 2 * v1 + 0) - element.variables(pOff + 2 * v0 + 0),
              element.variables(pOff + 2 * v2 + 0) - element.variables(pOff + 2 * v0 + 0),
              element.variables(pOff + 2 * v1 + 1) - element.variables(pOff + 2 * v0 + 1),
              element.variables(pOff + 2 * v2 + 1) - element.variables(pOff + 2 * v0 + 1);

        Eigen::Matrix<T, 2, 2> MrInv = Mr.inverse();
        T dA = T(0.5) * Mr.determinant();

        Eigen::Matrix<T, 3, 2> Ff = M * MrInv;

        Face f = mesh.face(f_idx);
        Eigen::Matrix3<T> L = computeShapeOperator_Adj<T>(geometry, element, f, ref_faces);

        T lam     = T(lambda_pf[f_idx]);
        T kap     = T(kappa_pf[f_idx]);
        T lam_sqr = lam * lam;
        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2<T>::Identity();
        Eigen::Matrix2<T> eps_b = (Ff.transpose() * L * Ff) - b_bar;

        T trM  = eps_b.trace();
        T trM2 = (eps_b * eps_b).trace();
        T Wb   = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3.0) / lam_sqr;
        return T(w_b) * Wb * dA;
      });

  return func;
}
