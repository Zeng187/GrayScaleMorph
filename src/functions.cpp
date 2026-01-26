#include "functions.h"

#include "simulation_utils.h"

#include <TinyAD/Utils/Helpers.hh>

using namespace geometrycentral::surface;

template <typename T, int R, int C>
inline T frob2(const Eigen::Matrix<T, R, C>& M)
{
  return (M.array() * M.array()).sum();
}



TinyAD::ScalarFunction<3, double, Vertex> simulationFunction(IntrinsicGeometryInterface& geometry,
                                                             const FaceData<Eigen::Matrix2d>& MrInv,
                                                             const FaceData<double>& lambda,
                                                             const VertexData<double>& kappa,
                                                             double E, double nu, double h,
                                                             double w_s, double w_b)
{


  SurfaceMesh& mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Set up function with 3D vertex positions as variables.
  TinyAD::ScalarFunction<3, double, Vertex> func = TinyAD::scalar_function<3>(mesh.vertices());

  // 1st fundamental form
  func.add_elements<3>(mesh.faces(),
                       [&, alpha, beta, E, nu, h, w_s, w_b, lambda, kappa](auto& element) -> TINYAD_SCALAR_TYPE(element) {
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

    // 第一基本型 a = F^T F
    Eigen::Matrix<T, 2, 2> a = F.transpose() * F;

    // 本征度量 a_bar = (lambda^2) I
    T lam = T(lambda[f]);
    T lam_sqr = lam * lam;

    // Green 应变 E = 0.5*(a_bar^{-1} a - I)
    Eigen::Matrix<T, 2, 2> Egreen = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();

    T trM = Egreen.trace();
    T trM2 = (Egreen * Egreen).trace();
    T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2);
    Ws = Ws * (1.0 / (lam_sqr * lam_sqr));
    return T(w_s) * Ws * dA;

  });



  func.add_elements<6>(
      mesh.faces(), [&, alpha, beta, E, nu, h, w_s, w_b, lambda, kappa](auto& element) -> TINYAD_SCALAR_TYPE(element) {
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

        // Compute deformation gradient
        Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f]);

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
          T theta = atan2(n.cross(nf).dot(e), e.norm() *  nf.dot(n));

          Eigen::Vector3<T> t = n.cross(e);

          // add edge contribution
          L += theta * t.normalized() * t.transpose();
        }
        L /= n.squaredNorm();

        T lam = T(lambda[f]);
        T lam_sqr = lam * lam;
        T kap = 0.0;
        for(Vertex v: f.adjacentVertices())
          kap += kappa[v] / 3;

        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2d ::Identity();

        // 本征度量 a_bar = (lambda^2) I
        Eigen::Matrix2<T> E = (Ff.transpose() * L * Ff) - b_bar;
        //Eigen::Matrix2<T> E = (((Ff.transpose() * L * Ff)) - b_bar);

        //std::cout << F.transpose() * L * F << '\n';


        T trM = E.trace();
        T trM2 = (E * E).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3);
        Wb = Wb * (1.0 / (lam_sqr * lam_sqr));
        return T(w_b) * Wb * dA;
      });

  return func;
}

TinyAD::ScalarFunction<3, double, Vertex> simulationFunction(IntrinsicGeometryInterface& geometry,
                                                           const geometrycentral::surface::VertexData<bool>& is_boundary_vertex,
                                                           const geometrycentral::surface::FaceData<bool>& is_boundary_face,
                                                           const geometrycentral::surface::FaceData<int>& boundary_ref_index,
                                                             const FaceData<Eigen::Matrix2d>& MrInv,
                                                             const FaceData<double>& lambda,
                                                             const FaceData<double>& kappa,
                                                             double E,
                                                             double nu,
                                                             double h,
                                                             double w_s,
                                                             double w_b)
{

  SurfaceMesh& mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Set up function with 3D vertex positions as variables.
  TinyAD::ScalarFunction<3, double, Vertex> func = TinyAD::scalar_function<3>(mesh.vertices());

  // 1st fundamental form
  func.add_elements<3>(
      mesh.faces(), [&, alpha, beta, E, nu, h, w_s, w_b, lambda, kappa](auto& element) -> TINYAD_SCALAR_TYPE(element) {
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

        // 第一基本型 a = F^T F
        Eigen::Matrix<T, 2, 2> a = F.transpose() * F;

        // 本征度量 a_bar = (lambda^2) I
        T lam = T(lambda[f]);
        T lam_sqr = lam * lam;

        // Green 应变 E = 0.5*(a_bar^{-1} a - I)
        Eigen::Matrix<T, 2, 2> Egreen = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();

        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();
        T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2);
        Ws = Ws * (1.0 / (lam_sqr * lam_sqr));
        return T(w_s) * Ws * dA;
      });

  func.add_elements<6>(
      mesh.faces(), [&, alpha, beta, E, nu, h, w_s, w_b, lambda, kappa](auto& element) -> TINYAD_SCALAR_TYPE(element) {
        // Evaluate element using either double or TinyAD::Double
        using T = TINYAD_SCALAR_TYPE(element);

        // Get variable 2D vertex positions
        Face f = element.handle;
        int f_id = f.getIndex();
        int rf_id = boundary_ref_index[f_id];
        f = mesh.face(rf_id);

        auto x0_idx = f.halfedge().vertex();
        auto x1_idx = f.halfedge().next().vertex();
        auto x2_idx = f.halfedge().next().next().vertex();
        Eigen::Vector3<T> x0 = element.variables(f.halfedge().vertex());
        Eigen::Vector3<T> x1 = element.variables(f.halfedge().next().vertex());
        Eigen::Vector3<T> x2 = element.variables(f.halfedge().next().next().vertex());

        Eigen::Matrix<T, 3, 2> M = TinyAD::col_mat(x1 - x0, x2 - x0);
        double dA = 0.5 / MrInv[f].determinant();

        // Compute deformation gradient
        Eigen::Matrix<T, 3, 2> Ff = M * (MrInv[f]);

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

        T lam = T(lambda[f]);
        T lam_sqr = lam * lam;
        T kap = T(kappa[f]);

        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2d ::Identity();

        // 本征度量 a_bar = (lambda^2) I
        Eigen::Matrix2<T> E = (Ff.transpose() * L * Ff) - b_bar;
        // Eigen::Matrix2<T> E = (((Ff.transpose() * L * Ff)) - b_bar);

        // std::cout << F.transpose() * L * F << '\n';

        T trM = E.trace();
        T trM2 = (E * E).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3);
        Wb = Wb * (1.0 / (lam_sqr * lam_sqr));
        return T(w_b) * Wb * dA;
      });

  return func;
}


TinyAD::ScalarFunction<3, double, Vertex> simulationFunction(IntrinsicGeometryInterface& geometry,
                                                             const FaceData<Eigen::Matrix2d>& MrInv,
                                                             const VertexData<double>& lambda,
                                                             const VertexData<double>& kappa,
                                                             double E,
                                                             double nu,
                                                             double h,
                                                             double w_s,
                                                             double w_b)
{

  SurfaceMesh& mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Set up function with 3D vertex positions as variables.
  TinyAD::ScalarFunction<3, double, Vertex> func = TinyAD::scalar_function<3>(mesh.vertices());

  // 1st fundamental form
  func.add_elements<3>(
      mesh.faces(), [&, alpha, beta, E, nu, h, w_s, w_b, lambda, kappa](auto& element) -> TINYAD_SCALAR_TYPE(element) {
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

        // 第一基本型 a = F^T F
        Eigen::Matrix<T, 2, 2> a = F.transpose() * F;

        // 本征度量 a_bar = (lambda^2) I
        T lam = 0.0;
        for(Vertex v: f.adjacentVertices())
          lam += lambda[v] / 3;
        T lam_sqr = lam * lam;

        // Green 应变 E = 0.5*(a_bar^{-1} a - I)
        Eigen::Matrix<T, 2, 2> Egreen = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();

        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();
        T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) ;
        Ws = Ws * (1.0 / (lam_sqr * lam_sqr));
        return T(w_s) * Ws * dA;
      });

  func.add_elements<6>(
      mesh.faces(), [&, alpha, beta, E, nu, h, w_s, w_b, lambda, kappa](auto& element) -> TINYAD_SCALAR_TYPE(element) {
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
        T lam_sqr = lam * lam;
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

        Eigen::Matrix2<T> b_bar = kap * Eigen::Matrix2d ::Identity();

        // 本征度量 a_bar = (lambda^2) I
        Eigen::Matrix2<T> E = (F.transpose() * L * F) - lam_sqr * b_bar;

        // std::cout << F.transpose() * L * F << '\n';

        T trM = E.trace();
        T trM2 = (E * E).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h *  T(1.0 / 3);
        Wb = Wb * (1.0 / (lam_sqr * lam_sqr));
        return T(w_b) * Wb * dA;
      });

  return func;
}




TinyAD::ScalarFunction<1, double, Eigen::Index> adjointFunction(IntrinsicGeometryInterface& geometry,
                                                                const Eigen::MatrixXi& F,
                                                                const FaceData<Eigen::Matrix2d>& MrInv,
                                                                const FaceData<double>& lambda,
                                                                double E,
                                                                double nu,
                                                                double h,
                                                                double w_s,
                                                                double w_b)
{
  SurfaceMesh& mesh = geometry.mesh;


  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Set up function with 3D vertex positions as variables.
  TinyAD::ScalarFunction<1, double, Eigen::Index> func =
      TinyAD::scalar_function<1>(TinyAD::range(3 * mesh.nVertices() + mesh.nVertices()));

  // 1st fundamental form
  func.add_elements<9>(TinyAD::range(F.rows()),
                       [&, alpha, beta, E, nu, h, w_s, w_b, lambda](auto& element) -> TINYAD_SCALAR_TYPE(element) {
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
                         T lam = T(lambda[f_idx]);
                         T lam_sqr = lam * lam;

                         // Green 应变 E = 0.5*(a_bar^{-1} a - I)
                         Eigen::Matrix<T, 2, 2> Egreen = a - lam_sqr * Eigen::Matrix<T, 2, 2>::Identity();

                         T trM = Egreen.trace();
                         T trM2 = (Egreen * Egreen).trace();
                         T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2);
                         Ws = Ws * 1.0 / (lam_sqr * lam_sqr);
                         return T(w_s) * Ws * dA;

                       });

  // 2nd fundamental form
  geometry.requireVertexIndices();

  func.add_elements<3 * 6 + 3>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, E, nu, h, w_s, w_b, lambda](auto& element) -> TINYAD_SCALAR_TYPE(element) {
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
        for(Halfedge he: f.adjacentHalfedges())
        {
          if(he.edge().isBoundary())
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

        T lam = T(lambda[f]);
        T lam_sqr = lam * lam;
        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2d ::Identity();

        // 本征度量 a_bar = (lambda^2) I

        Eigen::Matrix2<T> Egreen = (Ff.transpose() * L * Ff) - b_bar;

        //// std::cout << F.transpose() * L * F << '\n';

        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h *  T(1.0 / 3);
        Wb = Wb * 1.0 / (lam_sqr * lam_sqr);
        return T(w_b) * Wb * dA;

      });

  return func;
}


TinyAD::ScalarFunction<1, double, Eigen::Index> adjointFunction(IntrinsicGeometryInterface& geometry,
                                                                const Eigen::MatrixXi& F,
                                                                const FaceData<Eigen::Matrix2d>& MrInv,
                                                                const VertexData<double>& kappa,
                                                                double E,
                                                                double nu,
                                                                double h,
                                                                double w_s,
                                                                double w_b)
{
  SurfaceMesh& mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Set up function with 3D vertex positions as variables.
  TinyAD::ScalarFunction<1, double, Eigen::Index> func =
      TinyAD::scalar_function<1>(TinyAD::range(3 * mesh.nVertices() + mesh.nVertices()));

  // 1st fundamental form
  func.add_elements<6 + 6>(
      TinyAD::range(F.rows()),
                       [&, alpha, beta, E, nu, h, w_s, w_b](auto& element) -> TINYAD_SCALAR_TYPE(element) {
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
                         Ws = Ws * 1.0 / (lam_sqr * lam_sqr);
                         return T(w_s) * Ws * dA;
                       });

  // 2nd fundamental form
  geometry.requireVertexIndices();

  func.add_elements<3 * 6 + 3>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, E, nu, h, w_s, w_b, kappa](auto& element) -> TINYAD_SCALAR_TYPE(element) {
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
        for(Halfedge he: f.adjacentHalfedges())
        {
          if(he.edge().isBoundary())
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
        for(Vertex v: f.adjacentVertices())
          kap += kappa[v] / 3;

        Eigen::Matrix2<T> b_bar = lam_sqr * kap * Eigen::Matrix2d ::Identity();

        // 本征度量 a_bar = (lambda^2) I
        Eigen::Matrix2<T> Egreen = (Ff.transpose() * L * Ff) - b_bar;

        //// std::cout << F.transpose() * L * F << '\n';

        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h * T(1.0 / 3);
        Wb = Wb * 1.0 / (lam_sqr * lam_sqr);
        return T(w_b) * Wb * dA;
      });

  return func;
}

TinyAD::ScalarFunction<1, double, Eigen::Index> adjointFunction(IntrinsicGeometryInterface& geometry,
                                                                const Eigen::MatrixXi& F,
                                                                const FaceData<Eigen::Matrix2d>& MrInv,
                                                                double E,
                                                                double nu,
                                                                double h,
                                                                double w_s,
                                                                double w_b)
{
  SurfaceMesh& mesh = geometry.mesh;
  geometry.requireVertexIndices();

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  // Set up function with 3D vertex positions as variables.
  TinyAD::ScalarFunction<1, double, Eigen::Index> func =
      TinyAD::scalar_function<1>(TinyAD::range(3 * mesh.nVertices() + 2 * mesh.nVertices()));

  // 1st fundamental form
  func.add_elements<6 + 6>(TinyAD::range(F.rows()),
                       [&, alpha, beta, E, nu, h, w_s, w_b](auto& element) -> TINYAD_SCALAR_TYPE(element) {
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
                         //T lam = 1;

                         Eigen::Matrix<T, 2, 2> a_bar_inv = T(1.0 / (lam * lam)) * Eigen::Matrix<T, 2, 2>::Identity();

                         // Green 应变 E = 0.5*(a_bar^{-1} a - I)
                         Eigen::Matrix<T, 2, 2> Egreen = ((a_bar_inv * a) - Eigen::Matrix<T, 2, 2>::Identity());

                         T trM = Egreen.trace();
                         T trM2 = (Egreen * Egreen).trace();
                         T Ws = (T(0.5 * alpha) * trM * trM + T(beta) * trM2);
                         Ws = Ws * lam * lam;
                         return T(w_s) * Ws * dA;
                       });

  // 2nd fundamental form


  func.add_elements<3 * 6 + 6>(
      TinyAD::range(F.rows()),
      [&, alpha, beta, E, nu, h, w_s, w_b](auto& element) -> TINYAD_SCALAR_TYPE(element) {
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
        for(Halfedge he: f.adjacentHalfedges())
        {
          if(he.edge().isBoundary())
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

        Eigen::Vector3<T> kappa_f;
        kappa_f << element.variables(4 * mesh.nVertices() + F(f_idx, 0)),
            element.variables(4 * mesh.nVertices() + F(f_idx, 1)),
            element.variables(4 * mesh.nVertices() + F(f_idx, 2));
        T kap = (kappa_f(0) + kappa_f(1) + kappa_f(2)) / 3;

        Eigen::Matrix2<T> b_bar = kap * Eigen::Matrix2d ::Identity();

        // 本征度量 a_bar = (lambda^2) I
        Eigen::Matrix2<T> Egreen = (1.0 / (lam * lam)) * (Ff.transpose() * L * Ff) - b_bar;

        //// std::cout << F.transpose() * L * F << '\n';

        T trM = Egreen.trace();
        T trM2 = (Egreen * Egreen).trace();
        T Wb = (T(0.5 * alpha) * trM * trM + T(beta) * trM2) * h * h *  T(1.0 / 3);
        Wb = Wb * lam * lam;
        return T(w_b) * Wb * dA;
      });

  return func;
}

