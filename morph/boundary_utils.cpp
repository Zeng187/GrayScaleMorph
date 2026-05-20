#include "boundary_utils.h"

#include <spdlog/spdlog.h>
#include <Eigen/SVD>

#include <queue>

using namespace geometrycentral::surface;

// ---------------------------------------------------------------------------
// flatPlateAligned
// ---------------------------------------------------------------------------

Eigen::MatrixXd flatPlateAligned(const Eigen::MatrixXd& P,
                                  const Eigen::MatrixXd& V,
                                  const std::vector<int>& fixedVertexIdx)
{
    const Eigen::Index nV = P.rows();

    // Embed 2-D parameterisation as flat plate (z = 0)
    Eigen::MatrixXd V_flat(nV, 3);
    V_flat.leftCols(2) = P;
    V_flat.col(2).setZero();

    // Procrustes: find the rigid transform R, t that maps the 3 fixed
    // vertices from the flat plate to their target positions in V.
    Eigen::Matrix<double, 3, 3> src, dst;
    for (int i = 0; i < 3; ++i)
    {
        src.row(i) = V_flat.row(fixedVertexIdx[i]);
        dst.row(i) = V.row(fixedVertexIdx[i]);
    }

    const Eigen::Vector3d sc = src.colwise().mean();
    const Eigen::Vector3d dc = dst.colwise().mean();
    const Eigen::Matrix3d H  = (src.rowwise() - sc.transpose()).transpose()
                              * (dst.rowwise() - dc.transpose());

    auto svd = H.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
    if (R.determinant() < 0)
    {
        Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
        D(2, 2) = -1.0;
        R = svd.matrixV() * D * svd.matrixU().transpose();
    }
    const Eigen::Vector3d t = dc - R * sc;

    // Apply the rigid transform to every vertex
    for (Eigen::Index i = 0; i < nV; ++i)
        V_flat.row(i) = (R * V_flat.row(i).transpose() + t).transpose();

    // Force the 3 fixed vertices to their exact target positions so the
    // Procrustes residual (due to the flat->3D triangle distortion) does not
    // shift the pinned DOFs away from xTarget.
    for (int v : fixedVertexIdx)
        V_flat.row(v) = V.row(v);

    return V_flat;
}

// ---------------------------------------------------------------------------
// identifyBoundaryFaces
// ---------------------------------------------------------------------------

std::vector<bool> identifyBoundaryFaces(SurfaceMesh& mesh)
{
    const int nF = static_cast<int>(mesh.nFaces());
    std::vector<bool> is_boundary(nF, false);

    for (Face f : mesh.faces())
    {
        for (Halfedge he : f.adjacentHalfedges())
        {
            if (he.edge().isBoundary())
            {
                is_boundary[f.getIndex()] = true;
                break;
            }
        }
    }
    return is_boundary;
}

// ---------------------------------------------------------------------------
// buildRefFaces
// ---------------------------------------------------------------------------

std::vector<int> buildRefFaces(SurfaceMesh& mesh, std::vector<bool>& is_boundary)
{
    const int nF = static_cast<int>(mesh.nFaces());

    is_boundary = identifyBoundaryFaces(mesh);
    std::vector<int>  ref(nF, -1);

    // Seeds: interior faces reference themselves.
    std::queue<int> queue;
    for (int fi = 0; fi < nF; ++fi)
    {
        if (!is_boundary[fi])
        {
            ref[fi] = fi;
            queue.push(fi);
        }
    }

    // BFS on face dual graph (through non-boundary edges).
    while (!queue.empty())
    {
        int fi = queue.front();
        queue.pop();

        Face f = mesh.face(fi);
        for (Halfedge he : f.adjacentHalfedges())
        {
            if (he.edge().isBoundary())
                continue;
            int ni = he.twin().face().getIndex();
            if (ref[ni] < 0)
            {
                // Propagate: boundary face inherits the same ref as its source.
                ref[ni] = ref[fi];
                queue.push(ni);
            }
        }
    }

    // Fallback for unreachable faces.
    int n_unreachable = 0;
    for (int fi = 0; fi < nF; ++fi)
    {
        if (ref[fi] < 0)
        {
            ref[fi] = fi;  // self-reference (best effort)
            ++n_unreachable;
        }
    }

    int n_boundary = 0;
    for (int fi = 0; fi < nF; ++fi)
        if (is_boundary[fi])
            ++n_boundary;

    if (n_boundary > 0)
        spdlog::info("buildRefFaces: {} boundary faces mapped to interior references.", n_boundary);
    if (n_unreachable > 0)
        spdlog::warn("buildRefFaces: {} faces unreachable from any interior face.", n_unreachable);

    return ref;
}
