#include"output.hpp"

#include <igl/writeOBJ.h>

void write_output_obj(const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    Eigen::MatrixXd Vout(V.rows(), 3);
    Vout.col(0) = V.col(0);
    Vout.col(1) = V.col(1);
    Vout.col(2).setZero();
    igl::writeOBJ(filename, Vout, F);
}


void write_output_vtk(const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{

    std::ofstream out_file(filename + ".vtk", std::ofstream::out);
    if (!out_file)
    {
        throw std::runtime_error("Error: Problem creating or opening vtk output file.");
    }


    const int v_n = V.rows();
    const int f_n = F.rows();

    const int v_dim = V.cols();

    out_file << std::scientific << std::setprecision(14) << "# vtk DataFile Version 4.2" << "\n";

    // Continue with rest of preamble and then output the relevant data to file.
    out_file << "\n"
        << "ASCII" << "\n"
        << "DATASET POLYDATA" << "\n"
        << "POINTS " << v_n << " double" << "\n";

    for (int n = 0; n < v_n; ++n)
    {
        if (v_dim == 2)
        {
            out_file << V(n, 0) << " " << V(n, 1) << "\n";
        }
        else
        {
            out_file << V(n, 0) << " " << V(n, 1) << " " << V(n, 2) << "\n";
        }
    }

    out_file << "POLYGONS " << f_n << " " << 4 * f_n << "\n";

    for (int n = 0; n < f_n; ++n)
    {
        out_file << "3 " << F(n, 0) << " " << F(n, 1) << " " << F(n, 2) << "\n";
    }


}


void write_output_vtk(const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& a_comps,
    const Eigen::MatrixXd& b_comps)
{

    std::ofstream out_file(filename + ".vtk", std::ofstream::out);
    if (!out_file)
    {
        throw std::runtime_error("Error: Problem creating or opening vtk output file.");
    }

    const int v_n = V.rows();
    const int f_n = F.rows();

    const int v_dim = V.cols();

    out_file << std::scientific << std::setprecision(14) << "# vtk DataFile Version 4.2" << "\n";

    // Continue with rest of preamble and then output the relevant data to file.
    out_file << "\n"
        << "ASCII" << "\n"
        << "DATASET POLYDATA" << "\n"
        << "POINTS " << v_n << " double" << "\n";

    for (int n = 0; n < v_n; ++n)
    {
        if (v_dim == 2)
        {
            out_file << V(n, 0) << " " << V(n, 1) << "\n";
        }
        else
        {
            out_file << V(n, 0) << " " << V(n, 1) << " " << V(n, 2) << "\n";
        }
    }

    out_file << "POLYGONS " << f_n << " " << 4 * f_n << "\n";

    for (int n = 0; n < f_n; ++n)
    {
        out_file << "3 " << F(n, 0) << " " << F(n, 1) << " " << F(n, 2) << "\n";
    }

    out_file << "CELL_DATA " << f_n << "\n";


    // 写入参考 metric a_comps
    out_file << "VECTORS acomps_components double" << "\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << a_comps(t, 0) << " " << a_comps(t, 1) << " " << a_comps(t, 2) << "\n";
    }

    // 写入参考曲率 b_comps（vector）
    out_file << "VECTORS bcomps_components double" << "\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << b_comps(t, 0) << " " << b_comps(t, 1) << " " << b_comps(t, 2) << "\n";
    }
}


void write_output_vtk(const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& lambda,
    const Eigen::VectorXd& kappa,
    const Eigen::MatrixXd& a_comps,
    const Eigen::MatrixXd& b_comps)
{
    std::ofstream out_file(filename + ".vtk", std::ofstream::out);
    if (!out_file)
    {
        throw std::runtime_error("Error: Problem creating or opening vtk output file.");
    }

    const int v_n = V.rows();
    const int f_n = F.rows();

    const int v_dim = V.cols();

    out_file << std::scientific << std::setprecision(14) << "# vtk DataFile Version 4.2" << "\n";

    // Continue with rest of preamble and then output the relevant data to file.
    out_file << "\n"
        << "ASCII" << "\n"
        << "DATASET POLYDATA" << "\n"
        << "POINTS " << v_n << " double" << "\n";

    for (int n = 0; n < v_n; ++n)
    {
        if (v_dim == 2)
        {
            out_file << V(n, 0) << " " << V(n, 1) << "\n";
        }
        else
        {
            out_file << V(n, 0) << " " << V(n, 1) << " " << V(n, 2) << "\n";
        }
    }

    out_file << "POLYGONS " << f_n << " " << 4 * f_n << "\n";

    for (int n = 0; n < f_n; ++n)
    {
        out_file << "3 " << F(n, 0) << " " << F(n, 1) << " " << F(n, 2) << "\n";
    }

    out_file << "CELL_DATA " << f_n << "\n";

    // lambda（1 component）
    out_file << "SCALARS lambda double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << lambda(t) << "\n";
    }

    // lambda（1 component）
    out_file << "SCALARS kappa double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << kappa(t) << "\n";
    }

    // 写入参考 metric a_comps
    out_file << "VECTORS acomps_components double" << "\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << a_comps(t, 0) << " " << a_comps(t, 1) << " " << a_comps(t, 2) << "\n";
    }

    // 写入参考曲率 b_comps（vector）
    out_file << "VECTORS bcomps_components double" << "\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << b_comps(t, 0) << " " << b_comps(t, 1) << " " << b_comps(t, 2) << "\n";
    }
}


void write_output_vtk(const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& lambda,
    const Eigen::VectorXd& kappa,
    const Eigen::VectorXd& lambda_diff,
    const Eigen::VectorXd& kappa_diff,
    const Eigen::VectorXd& Ws_density,
    const Eigen::VectorXd& Wb_density)
{
    std::ofstream out_file(filename + ".vtk", std::ofstream::out);
    if (!out_file)
    {
        throw std::runtime_error("Error: Problem creating or opening vtk output file.");
    }

    const int v_n = V.rows();
    const int f_n = F.rows();

    const int v_dim = V.cols();

    out_file << std::scientific << std::setprecision(14) << "# vtk DataFile Version 4.2" << "\n";

    // Continue with rest of preamble and then output the relevant data to file.
    out_file << "\n"
        << "ASCII" << "\n"
        << "DATASET POLYDATA" << "\n"
        << "POINTS " << v_n << " double" << "\n";

    for (int n = 0; n < v_n; ++n)
    {
        if (v_dim == 2)
        {
            out_file << V(n, 0) << " " << V(n, 1) << "\n";
        }
        else
        {
            out_file << V(n, 0) << " " << V(n, 1) << " " << V(n, 2) << "\n";
        }
    }

    out_file << "POLYGONS " << f_n << " " << 4 * f_n << "\n";

    for (int n = 0; n < f_n; ++n)
    {
        out_file << "3 " << F(n, 0) << " " << F(n, 1) << " " << F(n, 2) << "\n";
    }

    out_file << "CELL_DATA " << f_n << "\n";

    // lambda（1 component）
    out_file << "SCALARS lambda double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << lambda(t) << "\n";
    }

    // lambda（1 component）
    out_file << "SCALARS kappa double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << kappa(t) << "\n";
    }

    // lambda（1 component）
    out_file << "SCALARS lambda_diff double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << lambda_diff(t) << "\n";
    }

    // lambda（1 component）
    out_file << "SCALARS kappa_diff double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << kappa_diff(t) << "\n";
    }


    // Ws_density（1 component）
    out_file << "SCALARS stretching_energy_density double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << Ws_density(t) << "\n";
    }

    // lambda（1 component）
    out_file << "SCALARS bending_energy_density double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << Wb_density(t) << "\n";
    }

}


void write_output_vtk(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Eigen::VectorXd>& per_vertex_attributes_list,
    const std::vector<std::string>& per_vertex_attributes_string_list)
{
    if (per_vertex_attributes_list.size() != per_vertex_attributes_string_list.size())
    {
        throw std::runtime_error("Attributes list and names list size mismatch.");
    }

    std::ofstream out_file(filename, std::ofstream::out);
    if (!out_file)
    {
        throw std::runtime_error("Error: Problem creating or opening vtk output file.");
    }

    const int v_n = static_cast<int>(V.rows());
    const int f_n = static_cast<int>(F.rows());
    const int v_dim = static_cast<int>(V.cols());
    const int f_cols = static_cast<int>(F.cols());

    out_file << std::scientific << std::setprecision(14) << "# vtk DataFile Version 4.2" << "\n";
    out_file << "VTK conversion from Eigen matrices" << "\n";
    out_file << "ASCII" << "\n";
    out_file << "DATASET POLYDATA" << "\n";

    out_file << "POINTS " << v_n << " double" << "\n";
    for (int n = 0; n < v_n; ++n)
    {
        out_file << V(n, 0) << " " << V(n, 1) << " " << (v_dim == 3 ? V(n, 2) : 0.0) << "\n";
    }

    out_file << "POLYGONS " << f_n << " " << (f_cols + 1) * f_n << "\n";
    for (int n = 0; n < f_n; ++n)
    {
        out_file << f_cols;
        for (int c = 0; c < f_cols; ++c)
        {
            out_file << " " << F(n, c);
        }
        out_file << "\n";
    }

    if (!per_vertex_attributes_list.empty())
    {
        out_file << "POINT_DATA " << v_n << "\n";

        for (size_t i = 0; i < per_vertex_attributes_list.size(); ++i)
        {
            const Eigen::VectorXd& attr_data = per_vertex_attributes_list[i];
            const std::string& attr_name = per_vertex_attributes_string_list[i];

            if (attr_data.size() != v_n)
            {
                continue;
            }

            out_file << "SCALARS " << attr_name << " double 1" << "\n";
            out_file << "LOOKUP_TABLE default" << "\n";
            for (int t = 0; t < v_n; ++t)
            {
                out_file << attr_data(t) << "\n";
            }
        }
    }
}


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
    const Eigen::VectorXd& kappa_pf_r)
{
    std::ofstream out_file(filename + ".vtk", std::ofstream::out);
    if (!out_file)
    {
        throw std::runtime_error("Error: Problem creating or opening vtk output file.");
    }

    const int v_n = V.rows();
    const int f_n = F.rows();

    const int v_dim = V.cols();

    out_file << std::scientific << std::setprecision(14) << "# vtk DataFile Version 4.2" << "\n";

    // Continue with rest of preamble and then output the relevant data to file.
    out_file << "\n"
        << "ASCII" << "\n"
        << "DATASET POLYDATA" << "\n"
        << "POINTS " << v_n << " double" << "\n";

    for (int n = 0; n < v_n; ++n)
    {
        if (v_dim == 2)
        {
            out_file << V(n, 0) << " " << V(n, 1) << "\n";
        }
        else
        {
            out_file << V(n, 0) << " " << V(n, 1) << " " << V(n, 2) << "\n";
        }
    }

    out_file << "POLYGONS " << f_n << " " << 4 * f_n << "\n";

    for (int n = 0; n < f_n; ++n)
    {
        out_file << "3 " << F(n, 0) << " " << F(n, 1) << " " << F(n, 2) << "\n";
    }

    out_file << "POINT_DATA " << v_n << "\n";

    // lambda（1 component）
    out_file << "SCALARS lambda_target double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < v_n; ++t)
    {
        out_file << lambda_pv_t(t) << "\n";
    }

    out_file << "SCALARS lambda_result double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < v_n; ++t)
    {
        out_file << lambda_pv_r(t) << "\n";
    }

    // lambda（1 component）
    out_file << "SCALARS kappa_target double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < v_n; ++t)
    {
        out_file << kappa_pv_t(t) << "\n";
    }

    // lambda（1 component）
    out_file << "SCALARS kappa_result double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < v_n; ++t)
    {
        out_file << kappa_pv_r(t) << "\n";
    }

}


void write_output_vtk(const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& lambda,
    const Eigen::VectorXd& kappa,
    const Eigen::VectorXd& lambda_diff,
    const Eigen::VectorXd& kappa_diff,
    const Eigen::MatrixXd& a_comps,
    const Eigen::MatrixXd& b_comps)
{
    std::ofstream out_file(filename + ".vtk", std::ofstream::out);
    if (!out_file)
    {
        throw std::runtime_error("Error: Problem creating or opening vtk output file.");
    }

    const int v_n = V.rows();
    const int f_n = F.rows();

    const int v_dim = V.cols();

    out_file << std::scientific << std::setprecision(14) << "# vtk DataFile Version 4.2" << "\n";

    // Continue with rest of preamble and then output the relevant data to file.
    out_file << "\n"
        << "ASCII" << "\n"
        << "DATASET POLYDATA" << "\n"
        << "POINTS " << v_n << " double" << "\n";

    for (int n = 0; n < v_n; ++n)
    {
        if (v_dim == 2)
        {
            out_file << V(n, 0) << " " << V(n, 1) << "\n";
        }
        else
        {
            out_file << V(n, 0) << " " << V(n, 1) << " " << V(n, 2) << "\n";
        }
    }

    out_file << "POLYGONS " << f_n << " " << 4 * f_n << "\n";

    for (int n = 0; n < f_n; ++n)
    {
        out_file << "3 " << F(n, 0) << " " << F(n, 1) << " " << F(n, 2) << "\n";
    }

    out_file << "CELL_DATA " << f_n << "\n";

    // lambda（1 component）
    out_file << "SCALARS lambda double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << lambda(t) << "\n";
    }

    // lambda（1 component）
    out_file << "SCALARS kappa double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << kappa(t) << "\n";
    }

    // lambda（1 component）
    out_file << "SCALARS lambda_diff double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << lambda_diff(t) << "\n";
    }

    // lambda（1 component）
    out_file << "SCALARS kappa_diff double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << kappa_diff(t) << "\n";
    }

    // 写入参考 metric a_comps
    out_file << "VECTORS acomps_components double" << "\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << a_comps(t, 0) << " " << a_comps(t, 1) << " " << a_comps(t, 2) << "\n";
    }

    // 写入参考曲率 b_comps（vector）
    out_file << "VECTORS bcomps_components double" << "\n";
    for (int t = 0; t < f_n; ++t)
    {
        out_file << b_comps(t, 0) << " " << b_comps(t, 1) << " " << b_comps(t, 2) << "\n";
    }
}



void write_output_vtk(const std::string& filename,
    const std::string& headerText,
    const Eigen::MatrixXd& nodes,
    const Eigen::MatrixXi& triangulation,
    const Eigen::MatrixXd& abar_info,
    const Eigen::MatrixXd& bbar_info,
    const Eigen::VectorXd& ref_thicknesses,
    const Eigen::VectorXd& ref_shear_moduli,
    const Eigen::VectorXi& tri_tags,
    const Eigen::VectorXi& constraint_indicators,
    const Eigen::VectorXi& node_tags)
{
    const int numNodes = static_cast<int>(nodes.rows());
    const int numTris = static_cast<int>(triangulation.rows());

    // 基本检查，对齐 Python 断言
    if (!((nodes.cols() == 2 || nodes.cols() == 3) && triangulation.cols() == 3 &&
        constraint_indicators.size() == numNodes && node_tags.size() == numNodes))
        throw std::runtime_error("Bad inputs to write_VTK.");

    std::ofstream file(filename + ".vtk", std::ofstream::out);
    if (!file)
        throw std::runtime_error("Cannot open VTK file for writing.");

    // 浮点格式与精度：%.15e
    file.setf(std::ios::scientific);
    file << std::setprecision(15);

    // 1) 头和 POINTS（2D 自动补 z=0）
    file << "# vtk DataFile Version 2.0\n"
        << headerText << "\n"
        << "ASCII\n"
        << "DATASET POLYDATA\n"
        << "POINTS " << numNodes << " double\n";
    if (nodes.cols() == 2)
    {
        for (int i = 0; i < numNodes; ++i)
        {
            file << nodes(i, 0) << " " << nodes(i, 1) << " " << 0.0 << "\n";
        }
    }
    else
    { // 3D
        for (int i = 0; i < numNodes; ++i)
        {
            file << nodes(i, 0) << " " << nodes(i, 1) << " " << nodes(i, 2) << "\n";
        }
    }

    // 2) POLYGONS（每行：3 i j k）
    file << "POLYGONS " << numTris << " " << 4 * numTris << "\n";
    for (int t = 0; t < numTris; ++t)
    {
        file << 3 << " " << triangulation(t, 0) << " " << triangulation(t, 1) << " " << triangulation(t, 2) << "\n";
    }

    // 3) CELL_DATA + tri FIELD（顺序与名称必须匹配 Python）
    file << "CELL_DATA " << numTris << "\n"
        << "FIELD tri_quantities " << 5 << "\n";

    // abar_info: 3 components, numTris tuples
    file << "abar_info " << abar_info.cols() << " " << numTris << " double\n";
    for (int t = 0; t < numTris; ++t)
    {
        // (T x 3) 按行输出
        for (int c = 0; c < abar_info.cols(); ++c)
        {
            file << abar_info(t, c);
            file << (c + 1 < abar_info.cols() ? " " : "\n");
        }
    }

    // bbar_info
    file << "bbar_info " << bbar_info.cols() << " " << numTris << " double\n";
    for (int t = 0; t < numTris; ++t)
    {
        for (int c = 0; c < bbar_info.cols(); ++c)
        {
            file << bbar_info(t, c);
            file << (c + 1 < bbar_info.cols() ? " " : "\n");
        }
    }

    // ref_thicknesses（1 component）
    file << "ref_thicknesses 1 " << numTris << " double\n";
    for (int t = 0; t < numTris; ++t)
    {
        file << ref_thicknesses(t) << "\n";
    }

    // ref_shear_moduli（1 component）
    file << "ref_shear_moduli 1 " << numTris << " double\n";
    for (int t = 0; t < numTris; ++t)
    {
        file << ref_shear_moduli(t) << "\n";
    }

    // tri_tags（int）
    file << "tri_tags 1 " << numTris << " int\n";
    file.unsetf(std::ios::floatfield); // 用默认格式写 int
    for (int t = 0; t < numTris; ++t)
    {
        file << tri_tags(t) << "\n";
    }
    file.setf(std::ios::scientific);
    file << std::setprecision(15);

    // 4) POINT_DATA + node FIELD（与 Python 顺序一致）
    file << "POINT_DATA " << numNodes << "\n"
        << "FIELD node_quantities " << 2 << "\n";

    // constraint_indicators（int）
    file << "constraint_indicators 1 " << numNodes << " int\n";
    file.unsetf(std::ios::floatfield);
    for (int i = 0; i < numNodes; ++i)
    {
        file << constraint_indicators(i) << "\n";
    }

    // node_tags（int）
    file << "node_tags 1 " << numNodes << " int\n";
    for (int i = 0; i < numNodes; ++i)
    {
        file << node_tags(i) << "\n";
    }

    file.close();
}