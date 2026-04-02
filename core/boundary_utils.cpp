#include "boundary_utils.h"

#include <spdlog/spdlog.h>

#include <queue>

using namespace geometrycentral::surface;

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

// ---------------------------------------------------------------------------
// precomputeShapeOperators
// ---------------------------------------------------------------------------

FaceData<Eigen::Matrix3d>
precomputeShapeOperators(
    SurfaceMesh& mesh,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& ref_faces)
{
    Eigen::Matrix3d zero_mat = Eigen::Matrix3d::Zero();
    FaceData<Eigen::Matrix3d> L_data(mesh, zero_mat);

    for (Face f : mesh.faces())
    {
        const int fi = f.getIndex();

        // Use the reference face's neighborhood for L computation.
        Face rf = mesh.face(ref_faces[fi]);

        // Reference face edge vectors → face normal
        int i0 = rf.halfedge().vertex().getIndex();
        int i1 = rf.halfedge().next().vertex().getIndex();
        int i2 = rf.halfedge().next().next().vertex().getIndex();
        Eigen::Vector3d e01 = V.row(i1) - V.row(i0);
        Eigen::Vector3d e02 = V.row(i2) - V.row(i0);
        Eigen::Vector3d n = e01.cross(e02);

        Eigen::Matrix3d L = Eigen::Matrix3d::Zero();
        for (Halfedge he : rf.adjacentHalfedges())
        {
            if (he.edge().isBoundary())
                continue;

            int v0 = he.vertex().getIndex();
            int v1 = he.next().vertex().getIndex();
            int v2 = he.twin().next().next().vertex().getIndex();

            Eigen::Vector3d e_1 = V.row(v1) - V.row(v0);
            Eigen::Vector3d e_2 = V.row(v2) - V.row(v0);

            Eigen::Vector3d n_adj = e_2.cross(e_1);
            double theta = atan2(n.cross(n_adj).dot(e_1),
                                 e_1.norm() * n_adj.dot(n));

            Eigen::Vector3d t = n.cross(e_1);
            L += theta * t.normalized() * t.transpose();
        }
        L /= n.squaredNorm();

        L_data[f] = L;
    }

    return L_data;
}
