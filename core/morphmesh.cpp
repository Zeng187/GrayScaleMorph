#include "morphmesh.hpp"
#include "boundary_utils.h"
#include "material.hpp"
#include <igl/principal_curvature.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <spdlog/spdlog.h>

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

    t_layer_pv_1.resize(nV);
    t_layer_pv_2.resize(nV);
    t_layer_pf_1.resize(nF);
    t_layer_pf_2.resize(nF);
    t_layer_pv_1.setConstant(1);
    t_layer_pv_2.setConstant(1);
    t_layer_pf_1.setConstant(1);
    t_layer_pf_2.setConstant(1);

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
    double h,
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
        double lam_sqr = lam * lam;
        Eigen::Matrix<double, 3, 2> Fg = M * (MrInv[f]);
        Eigen::Matrix2d a_mat = Fg.transpose() * Fg;

        // Form A: eps_s = a - lambda^2 * I, W_stretch = (1/lambda^2) * W_StVK(eps_s) * dA
        Eigen::Matrix2d eps_s = a_mat - lam_sqr * Eigen::Matrix2d::Identity();
        double Ws = ComputeSVNorm(alpha, beta, eps_s) / lam_sqr;

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

        // Form A: eps_b = b - lambda^2 * kappa * I, W_bend = (h^2/3) * (1/lambda^2) * W_StVK(eps_b) * dA
        Eigen::Matrix2d b_mat = Fg.transpose() * L * Fg;
        Eigen::Matrix2d eps_b = b_mat - lam_sqr * kap * Eigen::Matrix2d::Identity();
        double Wb = ComputeSVNorm(alpha, beta, eps_b) * (h * h / 3.0) / lam_sqr;

        Ws_density[face_id] = Ws * dA;
        Wb_density[face_id] = Wb * dA;
    };
}


void Morphmesh::ComputeElasticEnergy(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
    const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
    const geometrycentral::surface::FaceData<double>& lambda,
    const geometrycentral::surface::VertexData<double>& kappa,
    double h,
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
        double lam_sqr = lam * lam;
        Eigen::Matrix<double, 3, 2> Fg = M * (MrInv[f]);
        Eigen::Matrix2d a_mat = Fg.transpose() * Fg;

        // Form A: eps_s = a - lambda^2 * I, W_stretch = (1/lambda^2) * W_StVK(eps_s) * dA
        Eigen::Matrix2d eps_s = a_mat - lam_sqr * Eigen::Matrix2d::Identity();
        double Ws = ComputeSVNorm(alpha, beta, eps_s) / lam_sqr;

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

        // Form A: eps_b = b - lambda^2 * kappa * I, W_bend = (h^2/3) * (1/lambda^2) * W_StVK(eps_b) * dA
        Eigen::Matrix2d b_mat = Fg.transpose() * L * Fg;
        Eigen::Matrix2d eps_b = b_mat - lam_sqr * kap * Eigen::Matrix2d::Identity();
        double Wb = ComputeSVNorm(alpha, beta, eps_b) * (h * h / 3.0) / lam_sqr;

        Ws_density[face_id] = Ws * dA;
        Wb_density[face_id] = Wb * dA;
    };
}


void Morphmesh::ComputeTLayersFromMorphophing(
    const Eigen::VectorXd& _lambda_pv,
    const Eigen::VectorXd& _kappa_pv,
    const M_Poly_Curve& _strain_curve,
    const M_Poly_Curve& _moduls_curve,
    const double & thickness,
    Eigen::VectorXd& t_layer_pv_1_,
    Eigen::VectorXd& t_layer_pv_2_
    )

{
    const double eps = 1e-6; // Keep away from boundaries to avoid numerical issues
    int N = _lambda_pv.size();

    for (int i = 0; i < N; ++i)
    {
        double lambda = _lambda_pv[i];
        double kappa = _kappa_pv[i];

        // Compute target strain values for each layer
        double s1 = lambda + (kappa * thickness) / 3.0 - 1;
        double s2 = lambda - (kappa * thickness) / 3.0 - 1;

        // Invert the material curve to find t parameter values
        double t1 = invert_poly(_strain_curve, s1);
        double t2 = invert_poly(_strain_curve, s2);

        t_layer_pv_1_[i] = t1;
        t_layer_pv_2_[i] = t2;
        // Clamp to valid range [eps, 1-eps] to avoid boundary issues
        // t_layer_pv_1_[i] = std::max(eps, std::min(1.0 - eps, t1));
        // t_layer_pv_2_[i] = std::max(eps, std::min(1.0 - eps, t2));
    }

}


void Morphmesh::ComputeMorphingFormTLayers(
    const Eigen::VectorXd& t_layer_pv_1_,
    const Eigen::VectorXd& t_layer_pv_2_,
    const M_Poly_Curve& _strain_curve,
    const M_Poly_Curve& _moduls_curve,
    const double & thickness,
    Eigen::VectorXd& _lambda_pv,
    Eigen::VectorXd& _kappa_pv)
{
    int nV = t_layer_pv_1_.size();

    for (int i = 0; i < nV; ++i)
    {
        double t1 = t_layer_pv_1_[i];
        double t2 = t_layer_pv_2_[i];

        // lambda = 0.5 * (strain(t1) + strain(t2))
        _lambda_pv[i] = compute_lamb_d(_strain_curve, t1, t2);
        // kappa = 1.5 * (strain(t1) - strain(t2)) / thickness
        _kappa_pv[i] = compute_curv_d(_strain_curve, thickness, t1, t2);
    }
}



void Morphmesh::SetMorphophing(
    const Eigen::VectorXd& _lambda_pv_t,
    const Eigen::VectorXd& _lambda_pf_t,
    const Eigen::VectorXd& _kappa_pv_t,
    const Eigen::VectorXd& _kappa_pf_t,
    Eigen::VectorXd& _lambda_pv_s,
    Eigen::VectorXd& _lambda_pf_s,
    Eigen::VectorXd& _kappa_pv_s,
    Eigen::VectorXd& _kappa_pf_s)
{
    _lambda_pv_s = _lambda_pv_t;
    _lambda_pf_s = _lambda_pf_t;
    _kappa_pv_s = _kappa_pv_t;
	_kappa_pf_s = _kappa_pf_t;
}

void Morphmesh::RestrictRange(Eigen::VectorXd& _data, double range_m, double range_M)
{
	_data = _data.cwiseMax(range_m).cwiseMin(range_M);
}



void Morphmesh::ComputeMorphophing(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const int &nV,
    const int &nF,
    const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
    Eigen::VectorXd& _lambda_pv,
    Eigen::VectorXd& _lambda_pf,
    Eigen::VectorXd& _kappa_pv,
    Eigen::VectorXd& _kappa_pf,
    Eigen::VectorXd* vertex_area_sum_out)
{
    using namespace Eigen;
    SurfaceMesh& mesh = geometry.mesh;

    // Ensure output accumulators start from zero regardless of caller state.
    _lambda_pv.setZero();
    _lambda_pf.setZero();
    _kappa_pv.setZero();
    _kappa_pf.setZero();

    // Cache per-face area (rest-domain) for weighting.
    Eigen::VectorXd face_area(nF);

    // =====================================================================
    // Pass 1: compute per-face (lambda, kappa) from deformation gradient
    //         and shape operator.  Boundary faces get incomplete L here;
    //         their kappa will be corrected by extrapolation below.
    // =====================================================================
    for (int face_id = 0; face_id < nF; ++face_id)
    {
        Face f = mesh.face(face_id);
        Eigen::Matrix2d MrInv_f = MrInv[f];

        // Face vertices
        Eigen::Vector3d x0 = V.row(f.halfedge().vertex().getIndex());
        Eigen::Vector3d x1 = V.row(f.halfedge().next().vertex().getIndex());
        Eigen::Vector3d x2 = V.row(f.halfedge().next().next().vertex().getIndex());

        Eigen::Matrix<double, 3, 2> M = TinyAD::col_mat(x1 - x0, x2 - x0);
        Eigen::MatrixXd Fg = M * MrInv_f;

        // Face normal (unnormalized)
        Eigen::Vector3d n = M.col(0).cross(M.col(1));

        // Shape operator via dihedral angles (boundary edges skipped)
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

            Eigen::Vector3d n_adj = e_2.cross(e_1);
            double theta = atan2(n.cross(n_adj).dot(e_1),
                                 e_1.norm() * n_adj.dot(n));

            Eigen::Vector3d t = n.cross(e_1);
            L += theta * t.normalized() * t.transpose();
        }
        L /= n.squaredNorm();

        // First fundamental form  a = Fg^T Fg
        Eigen::Matrix2d a_mat = Fg.transpose() * Fg;
        // Second fundamental form b = Fg^T L Fg
        Eigen::Matrix2d b_mat = Fg.transpose() * L * Fg;

        double lam = sqrt(0.5 * a_mat.trace());
        double kap = 0.5 * (a_mat.inverse() * b_mat).trace();

        _lambda_pf[face_id] = lam;
        _kappa_pf[face_id]  = kap;

        face_area[face_id] = 0.5 / MrInv_f.determinant();

        // Accumulate per-vertex lambda (kappa deferred until after extrapolation)
        for (Vertex v : f.adjacentVertices())
            _lambda_pv[v.getIndex()] += lam * face_area[face_id];
    }

    // =====================================================================
    // Pass 2: boundary kappa extrapolation via ref_faces lookup
    //
    // Boundary faces (≥1 boundary edge) have incomplete shape operators.
    // Replace their kappa with the kappa of their nearest interior reference
    // face (determined by BFS on the face dual graph).
    // =====================================================================
    std::vector<bool> is_boundary;
    std::vector<int> ref_faces = buildRefFaces(mesh, is_boundary);

    for (int fi = 0; fi < nF; ++fi)
    {
        if (ref_faces[fi] != fi)
            _kappa_pf[fi] = _kappa_pf[ref_faces[fi]];
    }

    // =====================================================================
    // Pass 3: per-vertex kappa (area-weighted average of corrected per-face)
    // =====================================================================
    for (int fi = 0; fi < nF; ++fi)
    {
        Face f = mesh.face(fi);
        double kap = _kappa_pf[fi];
        double area = face_area[fi];
        for (Vertex v : f.adjacentVertices())
            _kappa_pv[v.getIndex()] += kap * area;
    }

    // =====================================================================
    // Pass 4: normalize per-vertex values by total area
    // =====================================================================
    Eigen::VectorXd vertex_area_sum = Eigen::VectorXd::Zero(nV);
    for (int fi = 0; fi < nF; ++fi)
    {
        Face f = mesh.face(fi);
        for (Vertex v : f.adjacentVertices())
            vertex_area_sum[v.getIndex()] += face_area[fi];
    }

    for (auto vert : mesh.vertices())
    {
        int vid = vert.getIndex();
        _lambda_pv[vid] /= vertex_area_sum[vid];
        _kappa_pv[vid]  /= vertex_area_sum[vid];
    }

    if (vertex_area_sum_out != nullptr)
        *vertex_area_sum_out = vertex_area_sum;
}






void Morphmesh::ComputeDiff()
{
    lambda_pf_diff = lambda_pf_r - lambda_pf_t;
    lambda_pv_diff = lambda_pv_r - lambda_pv_t;
    kappa_pf_diff = kappa_pf_r - kappa_pf_t;
    kappa_pv_diff = kappa_pv_r - kappa_pv_t;

}

void Morphmesh::ReassignPFFromPV(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& _lambda_pv,
    const Eigen::VectorXd& _kappa_pv,
    Eigen::VectorXd& _lambda_pf,
    Eigen::VectorXd& _kappa_pf)
{
    using namespace geometrycentral::surface;
    SurfaceMesh& mesh = geometry.mesh;

    for (Face f : mesh.faces())
    {
        int face_id = f.getIndex();

        // Get the three vertex indices of this face
        int x0_idx = f.halfedge().vertex().getIndex();
        int x1_idx = f.halfedge().next().vertex().getIndex();
        int x2_idx = f.halfedge().next().next().vertex().getIndex();

        // Average the per-vertex values to get per-face values
        _lambda_pf[face_id] = (_lambda_pv[x0_idx] + _lambda_pv[x1_idx] + _lambda_pv[x2_idx]) / 3.0;
        _kappa_pf[face_id] = (_kappa_pv[x0_idx] + _kappa_pv[x1_idx] + _kappa_pv[x2_idx]) / 3.0;
    }
}
