#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

void write_output_obj(const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F);


void write_output_vtk(const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F);

void write_output_vtk(const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& a_bar,
    const Eigen::MatrixXd& b_bar);

void write_output_vtk(const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& lambda,
    const Eigen::VectorXd& kappa,
    const Eigen::MatrixXd& a_bar,
    const Eigen::MatrixXd& b_bar);

void write_output_vtk(const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& lambda_pv_t,
    const Eigen::VectorXd& lambda_pf_t,
    const Eigen::VectorXd& kappa_pv_t,
    const Eigen::VectorXd& kappa_pf_t,
    const Eigen::VectorXd& lambda_pv_r,
    const Eigen::VectorXd& lambda_pf_r,
    const Eigen::VectorXd& kappa_pv_r,
    const Eigen::VectorXd& kappa_pf_r);

void write_output_vtk(const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& lambda,
    const Eigen::VectorXd& kappa,
    const Eigen::VectorXd& lambda_diff,
    const Eigen::VectorXd& kappa_diff,
    const Eigen::VectorXd& Ws_density,
    const Eigen::VectorXd& Wb_density);

void write_output_vtk(const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& lambda,
    const Eigen::VectorXd& kappa,
    const Eigen::VectorXd& lambda_diff,
    const Eigen::VectorXd& kappa_diff,
    const Eigen::MatrixXd& a_comps,
    const Eigen::MatrixXd& b_comps);

void write_output_vtk(const std::string& fileName,
    const std::string& headerText,
    const Eigen::MatrixXd& nodes,
    const Eigen::MatrixXi& triangulation,
    const Eigen::MatrixXd& abar_info,
    const Eigen::MatrixXd& bbar_info,
    const Eigen::VectorXd& ref_thicknesses,
    const Eigen::VectorXd& ref_shear_moduli,
    const Eigen::VectorXi& tri_tags,
    const Eigen::VectorXi& constraint_indicators,
    const Eigen::VectorXi& node_tags);


void write_output_vtk(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Eigen::VectorXd>& per_vertex_attributes_list,
    const std::vector<std::string>& per_vertex_attributes_string_list);

void write_output_vtk_perface(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Eigen::VectorXd>& per_face_attributes_list,
    const std::vector<std::string>& per_face_attributes_string_list);