
#include"morph_functions.hpp"

geometrycentral::surface::FaceData<Eigen::Matrix2d>
precomputeMrInv(geometrycentral::surface::ManifoldSurfaceMesh& mesh, const Eigen::MatrixXd& P, const Eigen::MatrixXi& F)
{
  using namespace geometrycentral;
  using namespace geometrycentral::surface;

  FaceData<Eigen::Matrix2d> rest_shapes(mesh);

  for(int i = 0; i < F.rows(); ++i)
  {
    // Get 2D vertex positions
    Eigen::Vector2d a = P.row(F(i, 0));
    Eigen::Vector2d b = P.row(F(i, 1));
    Eigen::Vector2d c = P.row(F(i, 2));
    Eigen::Matrix2d Mr = TinyAD::col_mat(b - a, c - a);

    // Save 2-by-2 matrix with edge vectors as colums
    rest_shapes[i] = Mr.inverse();
  }
  return rest_shapes;
}


geometrycentral::surface::FaceData<Eigen::MatrixXd>
precomputeM(geometrycentral::surface::ManifoldSurfaceMesh& mesh, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
  using namespace geometrycentral;
  using namespace geometrycentral::surface;

  FaceData<Eigen::MatrixXd> rest_shapes(mesh);

  for(int i = 0; i < F.rows(); ++i)
  {
    // Get 2D vertex positions
    Eigen::Vector3d a = V.row(F(i, 0));
    Eigen::Vector3d b = V.row(F(i, 1));
    Eigen::Vector3d c = V.row(F(i, 2));
    Eigen::Matrix<double,3,2> M = TinyAD::col_mat(b - a, c - a);

    // Save 2-by-2 matrix with edge vectors as colums
    rest_shapes[i] = M;
  }
  return rest_shapes;
}