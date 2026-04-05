#pragma once

#include <Eigen/Core>

/// Rigid (rotation + translation) Procrustes alignment.
///
/// Given two point sets with known vertex-to-vertex correspondence,
/// finds (R, t) minimizing  ||R * source + t - target||^2_F.
///
/// @param source  Nx3 source points.
/// @param target  Nx3 target points (same N, same vertex ordering).
/// @return        Nx3 aligned source points.
Eigen::MatrixXd rigidAlign(const Eigen::MatrixXd& source,
                           const Eigen::MatrixXd& target);
