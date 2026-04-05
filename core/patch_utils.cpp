#include "patch_utils.h"

#include <fstream>
#include <map>
#include <set>
#include <unordered_map>
#include <stdexcept>

PatchData extractPatch(const Eigen::MatrixXd& V,
                       const Eigen::MatrixXi& F,
                       const std::vector<int>& seg_id,
                       int patch_id)
{
    PatchData patch;

    // 1. Collect faces belonging to this patch
    for (int f = 0; f < (int)seg_id.size(); ++f) {
        if (seg_id[f] == patch_id) {
            patch.global_face_ids.push_back(f);
        }
    }

    // 2. Collect unique vertices used by these faces (sorted)
    std::set<int> vert_set;
    for (int gf : patch.global_face_ids) {
        for (int c = 0; c < 3; ++c) {
            vert_set.insert(F(gf, c));
        }
    }
    patch.global_vertex_ids.assign(vert_set.begin(), vert_set.end());

    // 3. Build global-to-local vertex mapping
    std::unordered_map<int, int> g2l;
    for (int i = 0; i < (int)patch.global_vertex_ids.size(); ++i) {
        g2l[patch.global_vertex_ids[i]] = i;
    }

    // 4. Build V_patch
    int nV_patch = (int)patch.global_vertex_ids.size();
    patch.V.resize(nV_patch, 3);
    for (int i = 0; i < nV_patch; ++i) {
        patch.V.row(i) = V.row(patch.global_vertex_ids[i]);
    }

    // 5. Build F_patch with re-indexed vertices
    int nF_patch = (int)patch.global_face_ids.size();
    patch.F.resize(nF_patch, 3);
    for (int i = 0; i < nF_patch; ++i) {
        int gf = patch.global_face_ids[i];
        for (int c = 0; c < 3; ++c) {
            patch.F(i, c) = g2l[F(gf, c)];
        }
    }

    return patch;
}

void linearSubdivide(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    const int nV_old = static_cast<int>(V.rows());
    const int nF_old = static_cast<int>(F.rows());
    const int nCols  = static_cast<int>(V.cols());

    std::map<std::pair<int,int>, int> edgeMid;
    int nextVid = nV_old;

    // Upper bound: each face has 3 edges, each shared by ≤2 faces
    Eigen::MatrixXd V_new(nV_old + 3 * nF_old, nCols);
    V_new.topRows(nV_old) = V;

    auto getMid = [&](int a, int b) -> int {
        auto key = std::make_pair(std::min(a, b), std::max(a, b));
        auto it = edgeMid.find(key);
        if (it != edgeMid.end()) return it->second;
        int mid = nextVid++;
        V_new.row(mid) = 0.5 * (V.row(a) + V.row(b));
        edgeMid[key] = mid;
        return mid;
    };

    Eigen::MatrixXi F_new(4 * nF_old, 3);
    for (int fi = 0; fi < nF_old; ++fi) {
        int v0 = F(fi, 0), v1 = F(fi, 1), v2 = F(fi, 2);
        int m01 = getMid(v0, v1);
        int m12 = getMid(v1, v2);
        int m20 = getMid(v2, v0);
        F_new.row(4 * fi + 0) << v0,  m01, m20;
        F_new.row(4 * fi + 1) << m01, v1,  m12;
        F_new.row(4 * fi + 2) << m20, m12, v2;
        F_new.row(4 * fi + 3) << m01, m12, m20;
    }

    V = V_new.topRows(nextVid);
    F = std::move(F_new);
}

std::vector<int> loadSegId(const std::string& path)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open seg_id file: " + path);
    }

    std::vector<int> seg_id;
    int val;
    while (ifs >> val) {
        seg_id.push_back(val);
    }
    return seg_id;
}
