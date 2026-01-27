#include "morphmesh.hpp"
#include <igl/principal_curvature.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>

Morphmesh::Morphmesh(const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& P,
    const Eigen::MatrixXi& F,
    double _E,
    double _nu)
    : E(_E), nu(_nu)
{
    init(V, P, F);
}

void Morphmesh::init(const Eigen::MatrixXd& V, const Eigen::MatrixXd& P, const Eigen::MatrixXi& F)
{
    nV = V.rows();
    nF = F.rows();
    nP = P.rows();

    assert(nV == nP);

    lambda_pv_s.resize(nV);
    lambda_pv_r.resize(nV);
    lambda_pv_t.resize(nV);
    lambda_pv_s.setConstant(1);
    lambda_pv_r.setConstant(1);
    lambda_pv_t.setConstant(1);
    lambda_pf_s.resize(nF);
    lambda_pf_r.resize(nF);
    lambda_pf_t.resize(nF);
    lambda_pf_s.setConstant(1);
    lambda_pf_r.setConstant(1);
    lambda_pf_t.setConstant(1);

    kappa_pv_s.resize(nV);
    kappa_pv_r.resize(nV);
    kappa_pv_t.resize(nV);
    kappa_pv_s.setConstant(0);
    kappa_pv_r.setConstant(0);
    kappa_pv_t.setConstant(0);
    kappa_pf_s.resize(nF);
    kappa_pf_r.resize(nF);
    kappa_pf_t.resize(nF);
    kappa_pf_s.setConstant(0);
    kappa_pf_r.setConstant(0);
    kappa_pf_t.setConstant(0);



    lambda_pv_diff.resize(nV);
    lambda_pf_diff.resize(nF);
    lambda_pv_diff.setConstant(0);
    lambda_pf_diff.setConstant(0);
    kappa_pv_diff.resize(nV);
    kappa_pf_diff.resize(nF);
    kappa_pv_diff.setConstant(0);
    kappa_pf_diff.setConstant(0);

}

using namespace geometrycentral::surface;


void Morphmesh::ComputeMorphophing(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<bool>& boundary_vertex_flags,
    const std::vector<bool>& boundary_face_flags,
    const std::vector<int>& boundary_ref_indices,
    const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
    Eigen::VectorXd& _lambda_pv,
    Eigen::VectorXd& _lambda_pf,
    Eigen::VectorXd& _kappa_pv,
    Eigen::VectorXd& _kappa_pf)
{

    using namespace Eigen;
    SurfaceMesh& mesh = geometry.mesh;

    for (int face_id = 0; face_id < nF; ++face_id)
    {
        int ref_face_id = face_id;
        if (boundary_face_flags[face_id])
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
        for (Halfedge he : rf.adjacentHalfedges())
        {
            if (he.edge().isBoundary())
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


        double lam = sqrt(0.5 * a_mat.trace());
        _lambda_pf[face_id] = lam;

        double kap = 0.5 * (a_mat.inverse() * b_mat).trace();
        _kappa_pf[face_id] = kap;

        for (Vertex v : f.adjacentVertices())
        {
            int vert_id = v.getIndex();
            _lambda_pv[vert_id] += lam;
            _kappa_pv[vert_id] += kap;
        }
    }

    for (auto vert : mesh.vertices())
    {
        int nv = vert.degree();
        int vert_id = vert.getIndex();
        _kappa_pv[vert_id] /= nv;
        _lambda_pv[vert_id] /= nv;
    }

}


inline double frob2(const Eigen::Matrix2d& M)
{
    return (M.array() * M.array()).sum();
}


inline double ComputeSVNorm(const double& alpha, const double& beta, const Eigen::Matrix2d Egreen)
{
    //return frob2(Egreen);
    double trM = Egreen.trace();
    double trM2 = (Egreen * Egreen).trace();
    double Ws = (0.5 * alpha * trM * trM + beta * trM2);
    return Ws;
}

void Morphmesh::ComputeElasticEnergy(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
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


    for (Face f : mesh.faces())
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
        for (Halfedge he : f.adjacentHalfedges())
        {
            if (he.edge().isBoundary())
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
        Eigen::Matrix2d b_bar = kap * Eigen::Matrix2d::Identity();
        Eigen::Matrix2d Egreen_b = b_mat - b_bar;
        double Wb = ComputeSVNorm(alpha, beta, Egreen_b);

        Ws_density[face_id] = Ws;
        Wb_density[face_id] = Wb;
    };


}


void Morphmesh::ComputeElasticEnergy(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
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

    for (Face f : mesh.faces())
    {
        int face_id = f.getIndex();
        auto x0_idx = f.halfedge().vertex().getIndex();
        auto x1_idx = f.halfedge().next().vertex().getIndex();
        auto x2_idx = f.halfedge().next().next().vertex().getIndex();

        double lam = lambda[f];
        double kap = (kappa[x0_idx] + kappa[x1_idx] + kappa[x2_idx]) / 3.0;

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
        for (Halfedge he : f.adjacentHalfedges())
        {
            if (he.edge().isBoundary())
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
        Eigen::Matrix2d b_bar = kap * Eigen::Matrix2d::Identity();
        Eigen::Matrix2d Egreen_b = b_mat - b_bar;
        double Wb = ComputeSVNorm(alpha, beta, Egreen_b);

        Ws_density[face_id] = Ws;
        Wb_density[face_id] = Wb;
    };
}


void Morphmesh::SetMorphophing(Eigen::VectorXd& _lambda_r,
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


void Morphmesh::ComputeElasticEnergyDensity() {



}

//void Morphmesh::ComputeActualMorphing(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
//                                       const Eigen::MatrixXd& V,
//                                       const Eigen::MatrixXd& P,
//                                       const Eigen::MatrixXi& F)
//{
//    // compute lambda by area
//  using namespace Eigen;
//SurfaceMesh& mesh = geometry.mesh;
//geometry.requireFaceAreas(); // 确保面面积已计算
//
//  Eigen::VectorXd lambda_faces;
//  lambda_faces.resize(nF);
//  for(int face_id = 0; face_id < nF; ++face_id)
//  {
//    Face f = mesh.face(face_id);
//
//    int x0_idx = f.halfedge().vertex().getIndex();
//    int x1_idx = f.halfedge().next().vertex().getIndex();
//    int x2_idx = f.halfedge().next().next().vertex().getIndex();
//
//    Eigen::Vector3d x0 = V.row(x0_idx);
//    Eigen::Vector3d x1 = V.row(x1_idx);
//    Eigen::Vector3d x2 = V.row(x2_idx);
//    double area_V = ((x2 - x0).cross(x1-x0)).norm();
//
//    Eigen::Vector2d u0 = P.row(x0_idx);
//    Eigen::Vector2d u1 = P.row(x1_idx);
//    Eigen::Vector2d u2 = P.row(x2_idx);
//    Eigen::Vector3d p0 = Eigen::Vector3d(u0(0),u0(1),0.0);
//    Eigen::Vector3d p1 = Eigen::Vector3d(u1(0),u1(1),0.0);
//    Eigen::Vector3d p2 = Eigen::Vector3d(u2(0),u2(1),0.0);
//    double area_P = ((p2 - p0).cross(p1 - p0)).norm();
//
//    lambda_faces[face_id] = sqrt(area_V / area_P);
//    //lambda_faces[face_id] = 1;
//    //lambda_a[face_id] = (area_V / area_P);
//
//  }
//
//  Eigen::VectorXd lambda_verts;
//  lambda_verts.resize(nV);
//  lambda_verts.setConstant(0.0);
//  Eigen::VectorXd vertex_area_sums = lambda_verts;
//  for(int face_id = 0; face_id < nF; ++face_id)
//  {
//    Face f = mesh.face(face_id);
//    double face_area = geometry.faceAreas[f];
//    for(Vertex v: f.adjacentVertices())
//    {
//      int v_idx = v.getIndex();
//      lambda_verts[v_idx] += lambda_faces[face_id] * face_area;
//      vertex_area_sums[v_idx] += face_area;
//    }
//  }
//
//  // 归一化
//  for(int v_idx = 0; v_idx < nV; ++v_idx)
//  {
//    if(vertex_area_sums[v_idx] > 1e-9)
//    {
//      lambda_verts[v_idx] /= vertex_area_sums[v_idx];
//    }
//  }
//    lambda_a = lambda_verts;
//    //lambda_t = lambda_faces;
//
//
//      Eigen::VectorXd k1_f;
//      Eigen::VectorXd k2_f;
//      Eigen::VectorXd H_f;
//      Eigen::VectorXd K_f;
//
//
//    // 1) 顶点主曲率与方向
//    Eigen::MatrixXd PD1, PD2;                           // directions (unused here)
//    Eigen::VectorXd PV1, PV2;                           // values: k_max, k_min at vertices
//    igl::principal_curvature(V, F, PD1, PD2, PV1, PV2); // 顶点值  :contentReference[oaicite:1]{index=1}
//
//
//
//    //    for(int vertex_id = 0; vertex_id < nV; ++vertex_id)
//    //{
//    //      bool isFaceAdjacentToBoundary = false;
//    //        Vertex v = mesh.vertex(vertex_id);
//    //      for(auto nv: v.adjacentVertices())
//    //      {
//    //        if(nv.isBoundary())
//    //        {
//    //          isFaceAdjacentToBoundary = true;
//    //          break;
//    //        }
//    //      }
//    //      if(!isFaceAdjacentToBoundary)
//    //      {
//    //          continue;
//    //      }
//    //      printf("vertex %d: H = %f, K = %f, is_bound: %d \n", vertex_id, (PV1[vertex_id] + PV2[vertex_id])*0.5f, PV1[vertex_id] * PV2[vertex_id],isFaceAdjacentToBoundary);
//    //}
//
//    kappa_a = 0.5 * (PV1 + PV2); // 平均曲率
//    //kappa_a.setConstant(1.0);
//    //kappa_a = PV2;               // 平均曲率
//    //for(int face_id = 0; face_id < nF; ++face_id)
//    //{
//    //
//    //  Face f = mesh.face(face_id);
//    //  for(Vertex v: f.adjacentVertices())
//    //  {
//    //    int v_idx = v.getIndex();
//    //    double kappa_v = abs (PV1[v_idx] * PV2[v_idx]);
//    //    kappa_a[v_idx] = sqrt(kappa_v);
//    //  }
//    //}
//
//
//    // printf("face_id %d: kappa = %f\n", face_id, _kappa[face_id]);
//
//
//
//  //kappa_t.setConstant(0.0);
//  //for(int face_id = 0; face_id < nF; ++face_id)
//  //  {
//  //    Face f = mesh.face(face_id);
//  //    for(Vertex v: f.adjacentVertices())
//  //    {
//  //      int v_idx = v.getIndex();
//  //      double kappa_v = kappa_a[v_idx];
//  //      kappa_t[face_id] += kappa_v;
//  //    }
//  //    kappa_t[face_id] /= 3.0;
//  //  }
//
//    //// 2) 可选：用法向对齐符号，避免面法向与顶点法向不一致导致的符号翻转
//    //Eigen::MatrixXd FN, VN;
//    //igl::per_face_normals(V, F, FN);
//    //igl::per_vertex_normals(V, F, VN);
//
//    //const int m = F.rows();
//    //k1_f.resize(m);
//    //k2_f.resize(m);
//    //H_f.resize(m);
//    //K_f.resize(m);
//
//    //for(int f = 0; f < m; ++f)
//    //{
//    //  Eigen::Vector3i tri = F.row(f);
//    //  double v1[3], v2[3];
//    //  for(int c = 0; c < 3; ++c)
//    //  {
//    //    int vi = tri[c];
//    //    double s = (VN.row(vi).dot(FN.row(f)) >= 0.0) ? 1.0 : -1.0; // 可选的符号统一
//    //    v1[c] = s * PV1[vi];
//    //    v2[c] = s * PV2[vi];
//    //  }
//    //  double k1 = (v1[0] + v1[1] + v1[2]) / 3.0; // 面上 k_max
//    //  double k2 = (v2[0] + v2[1] + v2[2]) / 3.0; // 面上 k_min
//    //  if(k2 > k1)
//    //    std::swap(k1, k2); // 保证 k1 ≥ k2
//
//    //  k1_f[f] = k1;
//    //  k2_f[f] = k2;
//    //  H_f[f] = 0.5 * (k1 + k2); // 平均曲率
//    //  K_f[f] = k1 * k2;         // 高斯曲率
//    //}
//
//  }

void Morphmesh::ComputeDiff()
{
    lambda_pf_diff = lambda_pf_r - lambda_pf_t;
    lambda_pv_diff = lambda_pv_r - lambda_pv_t;
    kappa_pf_diff = kappa_pf_r - kappa_pf_t;
    kappa_pv_diff = kappa_pv_r - kappa_pv_t;

}
