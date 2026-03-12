#include "patch_utils.h"

#include <fstream>
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
