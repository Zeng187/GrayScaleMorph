#include "rigid_align.h"

#include <Eigen/Dense>

Eigen::MatrixXd rigidAlign(const Eigen::MatrixXd& source,
                           const Eigen::MatrixXd& target)
{
    assert(source.rows() == target.rows());
    assert(source.cols() == 3 && target.cols() == 3);

    // 1. Centroids
    const Eigen::RowVector3d c_s = source.colwise().mean();
    const Eigen::RowVector3d c_t = target.colwise().mean();

    // 2. Centered point sets
    const Eigen::MatrixXd A = source.rowwise() - c_s;
    const Eigen::MatrixXd B = target.rowwise() - c_t;

    // 3. Cross-covariance matrix (3x3)
    const Eigen::Matrix3d H = A.transpose() * B;

    // 4. SVD → optimal rotation
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // Ensure proper rotation (det = +1), not reflection
    Eigen::Matrix3d R = V * U.transpose();
    if (R.determinant() < 0.0) {
        V.col(2) *= -1.0;
        R = V * U.transpose();
    }

    // 5. Translation
    const Eigen::RowVector3d t = c_t - (R * c_s.transpose()).transpose();

    // 6. Apply
    return (source * R.transpose()).rowwise() + t;
}
