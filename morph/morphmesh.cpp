#include "morphmesh.hpp"
#include "material.hpp"
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
    const std::vector<bool>& boundary_vertex_flags,
    const std::vector<bool>& boundary_face_flags,
    const std::vector<int>& boundary_ref_indices,
    const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
    Eigen::VectorXd& _lambda_pv,
    Eigen::VectorXd& _lambda_pf,
    Eigen::VectorXd& _kappa_pv,
    Eigen::VectorXd& _kappa_pf,
    Eigen::VectorXd* vertex_area_sum_out)
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

        // Compute face area for area-weighted averaging
        double face_area = 0.5 / MrInv_f.determinant();

        for (Vertex v : f.adjacentVertices())
        {
            int vert_id = v.getIndex();
            _lambda_pv[vert_id] += lam * face_area;
            _kappa_pv[vert_id] += kap * face_area;
        }
    }

    // Compute total area at each vertex for area-weighted averaging
    Eigen::VectorXd vertex_area_sum = Eigen::VectorXd::Zero(nV);
    for (int face_id = 0; face_id < nF; ++face_id)
    {
        Face f = mesh.face(face_id);
        Eigen::Matrix2d MrInv_f = MrInv[f];
        double face_area = 0.5 / MrInv_f.determinant();

        for (Vertex v : f.adjacentVertices())
        {
            int vert_id = v.getIndex();
            vertex_area_sum[vert_id] += face_area;
        }
    }

    // Normalize by total area at each vertex
    for (auto vert : mesh.vertices())
    {
        int vert_id = vert.getIndex();
        _kappa_pv[vert_id] /= vertex_area_sum[vert_id];
        _lambda_pv[vert_id] /= vertex_area_sum[vert_id];
    }

    // Store vertex_area_sum if requested
    if (vertex_area_sum_out != nullptr)
    {
        *vertex_area_sum_out = vertex_area_sum;
    }

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
