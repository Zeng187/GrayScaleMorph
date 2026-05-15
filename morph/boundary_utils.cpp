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
