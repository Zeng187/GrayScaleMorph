#include "simulation_utils.h"

#include <Eigen/Core>
#include <igl/colon.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include<igl/boundary_loop.h>

Eigen::SparseMatrix<double> projectionMatrix(const std::vector<int>& fixedIdx, int size)
{
  using namespace Eigen;

  VectorXi indices(size - fixedIdx.size());
  int k = 0;
  for(int i = 0; i < size; ++i)
  {
    if(!std::binary_search(fixedIdx.begin(), fixedIdx.end(), i))
    {
      indices(k) = i;
      ++k;
    }
  }

  SparseMatrix<double> P, Id(size, size);
  Id.setIdentity();

  igl::slice(Id, indices, 1, P);

  return P;
}

Eigen::SparseMatrix<double> buildHGN(const Eigen::VectorXd& masses,
                                     const Eigen::SparseMatrix<double>& P,
                                     const Eigen::SparseMatrix<double>& M_theta,
                                     const Eigen::SparseMatrix<double>& H)
{
  using namespace Eigen;

  int n = P.cols();
  int m = P.rows();
  int nTheta = M_theta.rows();

  // Mass matrix x (P * M * P')
  SparseMatrix<double> D(n, n);
  D.reserve(n);
  for(int i = 0; i < n; ++i)
    D.insert(i, i) = masses(i);
  D = (P * D * P.transpose()).eval();

  std::vector<Triplet<double>> v;
  for(int i = 0; i < D.outerSize(); i++)
    for(typename SparseMatrix<double>::InnerIterator it(D, i); it; ++it)
      v.emplace_back(it.row(), it.col(), it.value());

  // Mass matrix theta
  for(int i = 0; i < M_theta.outerSize(); i++)
    for(typename SparseMatrix<double>::InnerIterator it(M_theta, i); it; ++it)
      v.emplace_back(it.row() + m, it.col() + m, it.value());

  // P * (df / dx)' * P'
  SparseMatrix<double> A = (P * H.block(0, 0, n, n) * P.transpose()).eval();
  for(int i = 0; i < A.outerSize(); i++)
    for(typename SparseMatrix<double>::InnerIterator it(A, i); it; ++it)
    {
      v.emplace_back(it.row(), it.col() + m + nTheta, it.value());
      v.emplace_back(it.col() + m + nTheta, it.row(), it.value());
    }

  // (df / dθ)' * P'
  A = (H.block(n, 0, nTheta, n) * P.transpose()).eval();
  for(int i = 0; i < A.outerSize(); i++)
    for(typename SparseMatrix<double>::InnerIterator it(A, i); it; ++it)
    {
      v.emplace_back(it.row() + m, it.col() + m + nTheta, it.value());
      v.emplace_back(it.col() + m + nTheta, it.row() + m, it.value());
    }

  SparseMatrix<double> HGN(2 * m + nTheta, 2 * m + nTheta);
  HGN.setFromTriplets(v.begin(), v.end());

  return HGN;
}

void updateHGN(Eigen::SparseMatrix<double>& HGN,
               const Eigen::SparseMatrix<double>& P,
               const Eigen::SparseMatrix<double>& H)
{
  using namespace Eigen;

  int n = P.cols();
  int m = P.rows();
  int nTheta = H.rows() - n;

  // P * (df / dx)' * P'
  Eigen::SparseMatrix<double> A = (P * H.block(0, 0, n, n) * P.transpose()).eval();
  igl::slice_into(A, igl::colon<int>(0, m - 1), igl::colon<int>(m + nTheta, 2 * m + nTheta - 1), HGN);

  // (df / dθ)' * P'
  A = (H.block(n, 0, nTheta, n) * P.transpose()).eval();
  igl::slice_into(A, igl::colon<int>(m, m + nTheta - 1), igl::colon<int>(m + nTheta, 2 * m + nTheta - 1), HGN);

  HGN = SparseMatrix<double>(HGN.selfadjointView<Upper>());
}

std::vector<int> findCenterVertexIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F)
{
  int centerIdx = 0;
  double dist = (P.row(F(0, 0)) + P.row(F(0, 1)) + P.row(F(0, 2))).norm();
  for(int i = 0; i < F.rows(); ++i)
  {
    if((P.row(F(i, 0)) + P.row(F(i, 1)) + P.row(F(i, 2))).norm() < dist)
    {
      dist = (P.row(F(i, 0)) + P.row(F(i, 1)) + P.row(F(i, 2))).norm();
      centerIdx = i;
    }
  }

  // Fixed Indices
  std::vector<int> fixedIdx = {F(centerIdx, 0),         F(centerIdx, 1),         F(centerIdx, 2)};
  std::sort(fixedIdx.begin(), fixedIdx.end());

  return fixedIdx;
}


std::vector<int> findCenterFaceIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F)
{
  int centerIdx = 0;
  double dist = (P.row(F(0, 0)) + P.row(F(0, 1)) + P.row(F(0, 2))).norm();
  for(int i = 0; i < F.rows(); ++i)
  {
    if((P.row(F(i, 0)) + P.row(F(i, 1)) + P.row(F(i, 2))).norm() < dist)
    {
      dist = (P.row(F(i, 0)) + P.row(F(i, 1)) + P.row(F(i, 2))).norm();
      centerIdx = i;
    }
  }

  // Fixed Indices
  std::vector<int> fixedIdx = {3 * F(centerIdx, 0), 3 * F(centerIdx, 0) + 1, 3 * F(centerIdx, 0) + 2,
                               3 * F(centerIdx, 1), 3 * F(centerIdx, 1) + 1, 3 * F(centerIdx, 1) + 2,
                               3 * F(centerIdx, 2), 3 * F(centerIdx, 2) + 1, 3 * F(centerIdx, 2) + 2};
  std::sort(fixedIdx.begin(), fixedIdx.end());

  return fixedIdx;
}


std::vector<int> findCornerFaceIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F)
{

  std::vector<int> fixedIdx;

  std::vector<std::vector<int>> loops;
  igl::boundary_loop(F, loops);
  if(loops.empty())
  {
    std::cerr << "No boundary loop found!" << std::endl;
    return {};
  }

  const std::vector<int>& boundary_loop = loops[0];
  int n = (int)boundary_loop.size();

  // 2️⃣ 计算每个边界点的转角
  std::vector<std::pair<int, double>> angle_list;
  angle_list.reserve(n);

  for(int i = 0; i < n; ++i)
  {
    int prev = boundary_loop[(i - 1 + n) % n];
    int curr = boundary_loop[i];
    int next = boundary_loop[(i + 1) % n];

    Eigen::Vector3d v_prev = (P.row(prev) - P.row(curr)).normalized();
    Eigen::Vector3d v_next = (P.row(next) - P.row(curr)).normalized();

    double cos_angle = std::clamp(v_prev.dot(v_next), -1.0, 1.0);
    double angle = std::acos(cos_angle);
    angle_list.emplace_back(curr, angle);
  }

  std::sort(angle_list.begin(), angle_list.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
  int sharp_vertex = angle_list.front().first;

  double max_dist = -1.0;
  int farthest_vertex = sharp_vertex;
  Eigen::Vector3d p0 = P.row(sharp_vertex);

  for(int v: boundary_loop)
  {
    double dist = (P.row(v) - p0.transpose()).norm();
    if(dist > max_dist)
    {
      max_dist = dist;
      farthest_vertex = v;
    }
  }

  std::vector<int> corner_vertices = {sharp_vertex, farthest_vertex};
  fixedIdx.reserve(corner_vertices.size() * 3);
  for(int v: corner_vertices)
  {
    fixedIdx.push_back(3 * v);
    fixedIdx.push_back(3 * v + 1);
    fixedIdx.push_back(3 * v + 2);
  }

  std::sort(fixedIdx.begin(), fixedIdx.end());
  fixedIdx.erase(std::unique(fixedIdx.begin(), fixedIdx.end()), fixedIdx.end());

  return fixedIdx;
}


//std::vector<int> findCornerVertexIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F)
//{
//  // 1️⃣ 找出边界环
//  std::vector<std::vector<int>> loops;
//  igl::boundary_loop(F, loops);
//  if(loops.empty())
//  {
//    std::cerr << "No boundary loop found!\n";
//    return {-1, -1};
//  }
//
//  // 选最长的边界环（防止存在孔洞）
//  const auto& boundary_loop =
//      *std::max_element(loops.begin(), loops.end(), [](const auto& a, const auto& b) { return a.size() < b.size(); });
//
//  // 2️⃣ 随机选第一个边界点作为基准（或后续也可加角度判定）
//  int v0 = boundary_loop.front();
//  Eigen::Vector2d p0 = P.row(v0);
//
//  // 3️⃣ 找出与 v0 距离最远的顶点（定义为 corner1）
//  double max_dist = -1.0;
//  int farthest = v0;
//  for(int v: boundary_loop)
//  {
//    double dist = (P.row(v) - p0.transpose()).norm();
//    if(dist > max_dist)
//    {
//      max_dist = dist;
//      farthest = v;
//    }
//  }
//
//  int corner1 = v0;
//  int corner2 = farthest;
//
//  std::cout << "Corner vertices: " << corner1 << ", " << corner2 << std::endl;
//  return std::vector<int>{corner1, corner2};
//}



std::vector<int> findCornerVertexIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F)
{

if(P.cols() < 2 || P.rows() == 0)
  {
    // 返回空向量或抛出错误，取决于错误处理策略
    return {};
  }
  const Eigen::VectorXd X = P.col(0); // x 坐标列
  const Eigen::VectorXd Y = P.col(1); // y 坐标列
  const int N = P.rows();

  const double epsilon = 1e-6;


  // 找到 min/max x 和 y 坐标
  double min_x = X.minCoeff();
  double max_x = X.maxCoeff();
  double min_y = Y.minCoeff();
  double max_y = Y.maxCoeff();

  // 初始化结果索引：P00, P01, P11, P10
  std::vector<int> corner_indices(4, -1);

  // 初始化用于查找 y 极值的变量
  double min_y_on_min_x = std::numeric_limits<double>::max();
  double max_y_on_min_x = std::numeric_limits<double>::lowest();
  double min_y_on_max_x = std::numeric_limits<double>::max();
  double max_y_on_max_x = std::numeric_limits<double>::lowest();

  // 第一次遍历：找到在 min_x 和 max_x 上的 y 极值
  for(int i = 0; i < N; ++i)
  {
    double current_x = X(i);
    double current_y = Y(i);

    // A. 落在 min_x 上的点
    if(std::abs(current_x - min_x) < epsilon)
    {
      if(current_y < min_y_on_min_x)
      {
        min_y_on_min_x = current_y;
      }
      if(current_y > max_y_on_min_x)
      {
        max_y_on_min_x = current_y;
      }
    }

    // B. 落在 max_x 上的点
    if(std::abs(current_x - max_x) < epsilon)
    {
      if(current_y < min_y_on_max_x)
      {
        min_y_on_max_x = current_y;
      }
      if(current_y > max_y_on_max_x)
      {
        max_y_on_max_x = current_y;
      }
    }
    // 注意：这里不需要 else，因为我们只关心极值 x 上的点
  }

  // 第二次遍历：找到匹配的索引
  // 注意：如果有多个点满足条件，我们只需返回其中一个。
  for(int i = 0; i < N; ++i)
  {
    double current_x = X(i);
    double current_y = Y(i);

    // 1. P00: min_x, 且 y 为 min_y_on_min_x
    if(corner_indices[0] == -1 && std::abs(current_x - min_x) < epsilon &&
       std::abs(current_y - min_y_on_min_x) < epsilon)
    {
      corner_indices[0] = i;
    }

    // 2. P01: min_x, 且 y 为 max_y_on_min_x
    if(corner_indices[1] == -1 && std::abs(current_x - min_x) < epsilon &&
       std::abs(current_y - max_y_on_min_x) < epsilon)
    {
      corner_indices[1] = i;
    }

    // 3. P11: max_x, 且 y 为 max_y_on_max_x
    if(corner_indices[2] == -1 && std::abs(current_x - max_x) < epsilon &&
       std::abs(current_y - max_y_on_max_x) < epsilon)
    {
      corner_indices[2] = i;
    }

    // 4. P10: max_x, 且 y 为 min_y_on_max_x
    if(corner_indices[3] == -1 && std::abs(current_x - max_x) < epsilon &&
       std::abs(current_y - min_y_on_max_x) < epsilon)
    {
      corner_indices[3] = i;
    }

    // 优化：如果所有四个点都找到了，可以提前退出
    if(corner_indices[0] != -1 && corner_indices[1] != -1 && corner_indices[2] != -1 && corner_indices[3] != -1)
    {
      break;
    }
  }

  return corner_indices;
}