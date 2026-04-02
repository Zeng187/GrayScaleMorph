#pragma once

#include <Eigen/Core>
#include <vector>
#include <string>

struct PatchData {
    Eigen::MatrixXd V;                 // sub-mesh vertices (nV_patch x 3)
    Eigen::MatrixXi F;                 // sub-mesh faces, re-indexed (nF_patch x 3)
    std::vector<int> global_face_ids;  // F_patch[i] corresponds to global face global_face_ids[i]
    std::vector<int> global_vertex_ids;// V_patch[i] corresponds to global vertex global_vertex_ids[i]
};

PatchData extractPatch(const Eigen::MatrixXd& V,
                       const Eigen::MatrixXi& F,
                       const std::vector<int>& seg_id,
                       int patch_id);

std::vector<int> loadSegId(const std::string& path);
