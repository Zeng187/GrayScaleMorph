#pragma once


#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <TinyAD/Support/GeometryCentral.hh>
#include <TinyAD/ScalarFunction.hh>
#include <geometrycentral/surface/intrinsic_geometry_interface.h>

// Forward declaration
struct M_Poly_Curve;

class Morphmesh
{
public:


    Morphmesh(const Eigen::MatrixXd& V, const Eigen::MatrixXd& P, const Eigen::MatrixXi& F, double _E, double _nu);
    void init(const Eigen::MatrixXd& V, const Eigen::MatrixXd& P, const Eigen::MatrixXi& F);

    static void ComputeMorphophing(geometrycentral::surface::IntrinsicGeometryInterface& geometry,
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
        Eigen::VectorXd& _kappa_pv ,
        Eigen::VectorXd& _kappa_pf );

    static void SetMorphophing(
        const Eigen::VectorXd& _lambda_pv_t,
        const Eigen::VectorXd& _lambda_pf_t,
        const Eigen::VectorXd& _kappa_pv_t,
        const Eigen::VectorXd& _kappa_pf_t,
        Eigen::VectorXd& _lambda_pv_s,
        Eigen::VectorXd& _lambda_pf_s,
        Eigen::VectorXd&  _kappa_pv_s,
        Eigen::VectorXd&  _kappa_pf_s );


    static void ComputeTLayersFromMorphophing(
        const Eigen::VectorXd& _lambda_pv,
        const Eigen::VectorXd& _kappa_pv,
        const M_Poly_Curve& _strain_curve,
        const M_Poly_Curve& _moduls_curve,
        const double & thickness,
        Eigen::VectorXd& t_layer_pv_1_,
        Eigen::VectorXd& t_layer_pv_2_);

    static void ComputeMorphingFormTLayers(
        const Eigen::VectorXd& t_layer_pv_1_,
        const Eigen::VectorXd& t_layer_pv_2_,
        const M_Poly_Curve& _strain_curve,
        const M_Poly_Curve& _moduls_curve,
        const double & thickness,
        Eigen::VectorXd& _lambda_pv,
        Eigen::VectorXd& _kappa_pv);

    static void RestrictRange(Eigen::VectorXd& _data, double range_m,double range_M);



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


    Eigen::VectorXd t_layer_pv_1;
    Eigen::VectorXd t_layer_pv_2;
    Eigen::VectorXd t_layer_pf_1;
    Eigen::VectorXd t_layer_pf_2;

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



