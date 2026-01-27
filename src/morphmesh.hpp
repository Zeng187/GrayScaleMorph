#pragma once


#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <TinyAD/Support/GeometryCentral.hh>
#include <TinyAD/ScalarFunction.hh>
#include <geometrycentral/surface/intrinsic_geometry_interface.h>


class Morphmesh
{
public:


    Morphmesh(const Eigen::MatrixXd& V, const Eigen::MatrixXd& P, const Eigen::MatrixXi& F, double _E, double _nu);
    void init(const Eigen::MatrixXd& V, const Eigen::MatrixXd& P, const Eigen::MatrixXi& F);

    void ComputeMorphophing(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const std::vector<bool>& boundary_vertex_flags,
        const std::vector<bool>& boundary_face_flags,
        const std::vector<int>& boundary_ref_indices,
        const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
        Eigen::VectorXd& _lambda_pv,
        Eigen::VectorXd& _lambda_pf,
        Eigen::VectorXd& _kappa_pv,
        Eigen::VectorXd& _kappa_pf);

    // calculate the morphophing paramters: lamba, kappa, a_mat, b_mat settings
    void ComputeMorphophing(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const std::vector<bool>& boundary_vertex_flags,
        const std::vector<bool>& boundary_face_flags,
        const std::vector<int>& boundary_ref_indices,
        const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
        Eigen::VectorXd& _lambda,
        Eigen::VectorXd& _kappa,
        Eigen::MatrixX3d& _a_comps,
        Eigen::MatrixX3d& _b_comps);

    void ComputeElasticEnergy(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
        const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
        const geometrycentral::surface::FaceData<double>& lambda,
        const geometrycentral::surface::FaceData<double>& kappa,
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        Eigen::VectorXd& Ws_density,
        Eigen::VectorXd& Wb_density);

    void ComputeElasticEnergy(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
        const geometrycentral::surface::FaceData<Eigen::Matrix2d>& MrInv,
        const geometrycentral::surface::FaceData<double>& lambda,
        const geometrycentral::surface::VertexData<double>& kappa,
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        Eigen::VectorXd& Ws_density,
        Eigen::VectorXd& Wb_density);

    void SetMorphophing(Eigen::VectorXd& _lambda_r,
        Eigen::VectorXd& _kappa_r,
        Eigen::MatrixX3d& _a_comps_r,
        Eigen::MatrixX3d& _b_comps_r,
        const Eigen::VectorXd& _lambda_s,
        const Eigen::VectorXd& _kappa_s,
        const Eigen::MatrixX3d& _a_comps_s,
        const Eigen::MatrixX3d& _b_comps_s);



    void ComputeElasticEnergyDensity();
    void ComputeDiff();

    // stretching anc cuEigen::MatrixX3d& _b_comps_r,rvature settings for initial setting(_s), simulation results(_r), target_surface(_t), actual(_a, by libigl and area check)
    Eigen::VectorXd lambda_pv_s;
    Eigen::VectorXd lambda_pv_r;
    Eigen::VectorXd lambda_pv_t;
    Eigen::VectorXd lambda_pf_s;
    Eigen::VectorXd lambda_pf_r;
    Eigen::VectorXd lambda_pf_t;


    Eigen::VectorXd kappa_pv_s;
    Eigen::VectorXd kappa_pv_r;
    Eigen::VectorXd kappa_pv_t;
    Eigen::VectorXd kappa_pf_s;
    Eigen::VectorXd kappa_pf_r;
    Eigen::VectorXd kappa_pf_t;



    Eigen::VectorXd lambda_pv_diff;
    Eigen::VectorXd lambda_pf_diff;
    Eigen::VectorXd kappa_pv_diff;
    Eigen::VectorXd kappa_pf_diff;


    Eigen::VectorXd theta_v_s;
    Eigen::VectorXd theta_v_r;
    Eigen::VectorXd theta_v_t;



    // the 1st & 2nd fundamental settings for initial setting(_s), simulation results(_r), target_surface(_t)
    //Eigen::MatrixX3d a_comps_s;
    //Eigen::MatrixX3d a_comps_r;
    //Eigen::MatrixX3d a_comps_t;
    //Eigen::MatrixX3d b_comps_s;
    //Eigen::MatrixX3d b_comps_r;
    //Eigen::MatrixX3d b_comps_t;

    //Eigen::VectorXd Area_Ratio_r;


    int nV;
    int nF;
    int nP;

    double E = 1.0;
    double nu = 0.5;

};