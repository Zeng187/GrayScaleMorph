#include "morphmesh.hpp"
#include <igl/principal_curvature.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>

MorphoMesh::MorphoMesh(const Eigen::MatrixXd& V,
                       const Eigen::MatrixXd& P,
                       const Eigen::MatrixXi& F,
                       double _E,
                       double _nu)
    : E(_E), nu(_nu)
{
  init(V, P, F);
}

void MorphoMesh::init(const Eigen::MatrixXd& V, const Eigen::MatrixXd& P, const Eigen::MatrixXi& F)
{
  nV = V.rows();
  nF = F.rows();
  nP = P.rows();

  assert(nV == nP);

  lambda_s.resize(nF);
  lambda_r.resize(nF);
  lambda_t.resize(nF);
  lambda_a.resize(nV);
  kappa_s.resize(nF);
  kappa_r.resize(nF);
  kappa_t.resize(nF);
  kappa_a.resize(nV);

  lambda_s.setConstant(1);
  lambda_r.setConstant(1);
  lambda_t.setConstant(1);
  lambda_a.setConstant(1);
  kappa_s.setConstant(0);
  kappa_r.setConstant(0);
  kappa_t.setConstant(0);
  kappa_a.setConstant(0);

  lambda_diff.resize(nF);
  kappa_diff.resize(nF);
  lambda_diff.setConstant(0);
  kappa_diff.setConstant(0);

  a_comps_s.resize(nF, 3);
  a_comps_r.resize(nF, 3);
  a_comps_t.resize(nF, 3);
  b_comps_s.resize(nF, 3);
  b_comps_r.resize(nF, 3);
  b_comps_t.resize(nF, 3);

  a_comps_s.setConstant(0);
  a_comps_r.setConstant(0);
  a_comps_t.setConstant(0);
  b_comps_s.setConstant(0);
  b_comps_r.setConstant(0);
  b_comps_t.setConstant(0);
}

using namespace geometrycentral::surface;

void MorphoMesh::ComputeMorphophing(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
                                    const Eigen::MatrixXd& V,
                                    const Eigen::MatrixXi& F,
                                    const std::vector<bool>& boundary_vertex_flags,
                                    const std::vector<bool>& boundary_face_flags,
                                    const std::vector<int>& boundary_ref_indices,
                                    const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
                                    Eigen::VectorXd& _lambda,
                                    Eigen::VectorXd& _kappa,
                                    Eigen::MatrixX3d& _a_comps,
                                    Eigen::MatrixX3d& _b_comps)
{

  using namespace Eigen;
  SurfaceMesh& mesh = geometry.mesh;

  for(int face_id = 0; face_id < nF; ++face_id)
  {
    int ref_face_id = face_id;
    if(boundary_face_flags[face_id])
      ref_face_id = boundary_ref_indices[face_id];

    Face f = mesh.face(face_id);
    Face rf = mesh.face(ref_face_id);


    Eigen::Matrix2d MrInv_f = MrInv[f];
    Eigen::Matrix2d MrInv_rf = MrInv[rf];

    int x0_idx_f = f.halfedge().vertex().getIndex();
    int x1_idx_f = f.halfedge().next().vertex().getIndex();
    int x2_idx_f = f.halfedge().next().next().vertex().getIndex();
    Eigen::Vector3d x0_f = V.row(x0_idx_f);
    Eigen::Vector3d x1_f = V.row(x1_idx_f);
    Eigen::Vector3d x2_f = V.row(x2_idx_f);

    int x0_idx_rf = rf.halfedge().vertex().getIndex();
    int x1_idx_rf = rf.halfedge().next().vertex().getIndex();
    int x2_idx_rf = rf.halfedge().next().next().vertex().getIndex();
    Eigen::Vector3d x0_rf = V.row(x0_idx_rf);
    Eigen::Vector3d x1_rf = V.row(x1_idx_rf);
    Eigen::Vector3d x2_rf = V.row(x2_idx_rf);

    Eigen::Matrix<double, 3, 2> M_f = TinyAD::col_mat(x1_f - x0_f, x2_f - x0_f);
    Eigen::Matrix<double, 3, 2> M_rf = TinyAD::col_mat(x1_rf - x0_rf, x2_rf - x0_rf);

    Eigen::MatrixXd Fg_f = M_f * MrInv_f;
    Eigen::MatrixXd Fg_rf = M_rf * MrInv_rf;



    Eigen::Vector3d tagent_0_rf = M_rf.col(0);
    Eigen::Vector3d tagent_1_rf = M_rf.col(1);
    auto n_rf = tagent_0_rf.cross(tagent_1_rf);


    Eigen::Matrix3d L = Eigen::Matrix3d::Zero();
    for(Halfedge he: rf.adjacentHalfedges())
    {
      if(he.edge().isBoundary())
      {
        continue;
      }

      auto v0 = he.vertex().getIndex();
      auto v1 = he.next().vertex().getIndex();
      auto v2 = he.twin().next().next().vertex().getIndex();

      Eigen::Vector3d e_1 = V.row(v1) - V.row(v0);
      Eigen::Vector3d e_2 = V.row(v2) - V.row(v0);

      // compute dihedral angle theta
      Eigen::Vector3d n_adj_rf = (e_2).cross(e_1);
      double theta = atan2(n_rf.cross(n_adj_rf).dot(e_1), e_1.norm() * n_adj_rf.dot(n_rf));

      Eigen::Vector3d t = n_rf.cross(e_1);

      // add edge contribution
      L += theta * t.normalized() * t.transpose();
    }
    L /= n_rf.squaredNorm();

    Eigen::Matrix2d a_mat = Fg_f.transpose() * Fg_f;
    Eigen::Matrix2d b_mat = Fg_rf.transpose() * L * Fg_rf;

    _a_comps.row(face_id) << a_mat(0, 0), (a_mat(1, 0) + a_mat(0, 1)) * 0.5, a_mat(1, 1);
    _b_comps.row(face_id) << b_mat(0, 0), (b_mat(1, 0) + b_mat(0, 1)) * 0.5, b_mat(1, 1);

    double lam = 0.5 * a_mat.trace();
    _lambda[face_id] = sqrt(lam);
    //_lambda[face_id] = (lam);

    double kap = 0.5 * (a_mat.inverse() * b_mat).trace();
    //double kap = 0.5 * (b_mat).trace();
    _kappa[face_id] = kap;
    
    }

}


inline double frob2(const Eigen::Matrix2d& M)
{
  return (M.array() * M.array()).sum();
}


inline double ComputeSVNorm(const double & alpha, const double & beta, const Eigen::Matrix2d Egreen) 
{
  //return frob2(Egreen);
  double trM = Egreen.trace();
  double trM2 = (Egreen * Egreen).trace();
  double Ws = (0.5 * alpha * trM * trM + beta * trM2);
  return Ws;
}

void MorphoMesh::ComputeElasticEnergy(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
                                      const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
                                      const geometrycentral::surface::FaceData<double>& lambda,
                                      const geometrycentral::surface::FaceData<double>& kappa,
                                      const Eigen::MatrixXd& V,
                                      const Eigen::MatrixXi& F,
                                      Eigen::VectorXd& Ws_density,
                                      Eigen::VectorXd& Wb_density)
{

  SurfaceMesh& mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));


  for(Face f: mesh.faces())
  {
      int face_id = f.getIndex();
      auto x0_idx = f.halfedge().vertex().getIndex();
      auto x1_idx = f.halfedge().next().vertex().getIndex();
      auto x2_idx = f.halfedge().next().next().vertex().getIndex();

      double lam = lambda[f];
      double kap = kappa[f];


      Eigen::Vector3d x0 = V.row(x0_idx);
      Eigen::Vector3d x1 = V.row(x1_idx);
      Eigen::Vector3d x2 = V.row(x2_idx);
      Eigen::Matrix<double, 3, 2> M = TinyAD::col_mat(x1 - x0, x2 - x0);

      double dA = 0.5 / MrInv[f].determinant();
      Eigen::Matrix<double, 3, 2> Fg = M * (MrInv[f]);
      Eigen::Matrix2d a_mat = Fg.transpose() * Fg;
      Eigen::Matrix2d a_bar_inv = (1.0 / (lam * lam)) * Eigen::Matrix2d::Identity();
      Eigen::Matrix2d Egreen_a = ((a_bar_inv * a_mat) - Eigen::Matrix2d::Identity());
      double Ws = ComputeSVNorm(alpha, beta, Egreen_a);   


      Eigen::Vector3d f_0 = M.col(0);
      Eigen::Vector3d f_1 = M.col(1);
      auto n = f_0.cross(f_1);

      Eigen::Matrix3d L = Eigen::Matrix3d::Zero();
      for(Halfedge he: f.adjacentHalfedges())
      {
        if(he.edge().isBoundary())
          continue;

        auto v0 = he.vertex().getIndex();
        auto v1 = he.next().vertex().getIndex();
        auto v2 = he.twin().next().next().vertex().getIndex();

        Eigen::Vector3d e_1 = V.row(v1) - V.row(v0);
        Eigen::Vector3d e_2 = V.row(v2) - V.row(v0);

        // compute dihedral angle theta
        Eigen::Vector3d nf = (e_2).cross(e_1);
        double theta = atan2(n.cross(nf).dot(e_1), e_1.norm() * nf.dot(n));

        Eigen::Vector3d t = n.cross(e_1);

        // add edge contribution
        L += theta * t.normalized() * t.transpose();
      }
      L /= n.squaredNorm();

      Eigen::Matrix2d b_mat = Fg.transpose() * L * Fg;
      Eigen::Matrix2d b_bar = kap * Eigen::Matrix2d ::Identity();   
      Eigen::Matrix2d Egreen_b = b_mat - b_bar;
      double Wb = ComputeSVNorm(alpha, beta, Egreen_b);

      Ws_density[face_id] = Ws;
      Wb_density[face_id] = Wb;
  };


}


void MorphoMesh::ComputeElasticEnergy(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
                                      const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
                                      const geometrycentral::surface::FaceData<double>& lambda,
                                      const geometrycentral::surface::VertexData<double>& kappa,
                                      const Eigen::MatrixXd& V,
                                      const Eigen::MatrixXi& F,
                                      Eigen::VectorXd& Ws_density,
                                      Eigen::VectorXd& Wb_density)
{

  SurfaceMesh& mesh = geometry.mesh;

  const double alpha = E * nu / (1 - nu * nu);
  const double beta = E / (2 * (1 + nu));

  for(Face f: mesh.faces())
  {
    int face_id = f.getIndex();
    auto x0_idx = f.halfedge().vertex().getIndex();
    auto x1_idx = f.halfedge().next().vertex().getIndex();
    auto x2_idx = f.halfedge().next().next().vertex().getIndex();

    double lam = lambda[f];
    double kap = (kappa[x0_idx] + kappa[x1_idx] + kappa[x2_idx])/3.0;

    Eigen::Vector3d x0 = V.row(x0_idx);
    Eigen::Vector3d x1 = V.row(x1_idx);
    Eigen::Vector3d x2 = V.row(x2_idx);
    Eigen::Matrix<double, 3, 2> M = TinyAD::col_mat(x1 - x0, x2 - x0);

    double dA = 0.5 / MrInv[f].determinant();
    Eigen::Matrix<double, 3, 2> Fg = M * (MrInv[f]);
    Eigen::Matrix2d a_mat = Fg.transpose() * Fg;
    Eigen::Matrix2d a_bar_inv = (1.0 / (lam * lam)) * Eigen::Matrix2d::Identity();
    Eigen::Matrix2d Egreen_a = ((a_bar_inv * a_mat) - Eigen::Matrix2d::Identity());
    double Ws = ComputeSVNorm(alpha, beta, Egreen_a);

    Eigen::Vector3d f_0 = M.col(0);
    Eigen::Vector3d f_1 = M.col(1);
    auto n = f_0.cross(f_1);

    Eigen::Matrix3d L = Eigen::Matrix3d::Zero();
    for(Halfedge he: f.adjacentHalfedges())
    {
      if(he.edge().isBoundary())
        continue;

      auto v0 = he.vertex().getIndex();
      auto v1 = he.next().vertex().getIndex();
      auto v2 = he.twin().next().next().vertex().getIndex();

      Eigen::Vector3d e_1 = V.row(v1) - V.row(v0);
      Eigen::Vector3d e_2 = V.row(v2) - V.row(v0);

      // compute dihedral angle theta
      Eigen::Vector3d nf = (e_2).cross(e_1);
      double theta = atan2(n.cross(nf).dot(e_1), e_1.norm() * nf.dot(n));

      Eigen::Vector3d t = n.cross(e_1);

      // add edge contribution
      L += theta * t.normalized() * t.transpose();
    }
    L /= n.squaredNorm();

    Eigen::Matrix2d b_mat = Fg.transpose() * L * Fg;
    Eigen::Matrix2d b_bar = kap * Eigen::Matrix2d ::Identity();
    Eigen::Matrix2d Egreen_b = b_mat - b_bar;
    double Wb = ComputeSVNorm(alpha, beta, Egreen_b);

    Ws_density[face_id] = Ws;
    Wb_density[face_id] = Wb;
  };
}


void MorphoMesh::SetMorphophing(Eigen::VectorXd& _lambda_r,
    Eigen::VectorXd& _kappa_r,
    Eigen::MatrixX3d& _a_comps_r,
    Eigen::MatrixX3d& _b_comps_r,
    const Eigen::VectorXd& _lambda_s,
    const Eigen::VectorXd& _kappa_s,
    const Eigen::MatrixX3d& _a_comps_s,
    const Eigen::MatrixX3d& _b_comps_s)
{
    _lambda_r = _lambda_s;
    _kappa_r = _kappa_s;
    _a_comps_r = _a_comps_s;
    _b_comps_r = _b_comps_s;
}


void MorphoMesh::ComputeElasticEnergyDensity() {



}


void MorphoMesh::ComputeDiff()
{
    lambda_diff = lambda_r - lambda_t;
    kappa_diff = kappa_r - kappa_t;

}
