
#include <igl/readOBJ.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include <spdlog/spdlog.h>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include "config.hpp"
#include "patch_utils.h"
#include "material.hpp"
#include "parameterization.h"
#include "morphmesh.hpp"
#include "morph_functions.hpp"

int main(int argc, char* argv[])
{
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    Config config("cfg.json");

    std::string mesh_path     = config.ModelSetting.InputPath + config.ModelSetting.ModelName + config.ModelSetting.Postfix;
    std::string segid_path    = config.ResourceSetting.SegmentPath + config.ModelSetting.ModelName + "/seg_id.txt";
    std::string material_path = config.ResourceSetting.MaterialPath;
    std::string output_path   = config.OutputSetting.MetricsPath + config.ModelSetting.ModelName + "/metrics.txt";
    double platewidth         = config.RuntimeSetting.Platewidth;

    std::filesystem::create_directories(config.OutputSetting.MetricsPath + config.ModelSetting.ModelName);

    // 1. Load mesh
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::readOBJ(mesh_path, V, F)) {
        spdlog::error("Cannot read mesh: {}", mesh_path);
        return 1;
    }
    int nF_global = F.rows();
    spdlog::info("Loaded mesh: {} vertices, {} faces", V.rows(), F.rows());

    // 2. Scale V to platewidth
    double scaleFactor = platewidth / (V.colwise().maxCoeff() - V.colwise().minCoeff()).maxCoeff();
    V *= scaleFactor;

    // 3. Load material
    ActiveComposite ac(material_path);
    ac.ComputeMaterialCurve();
    ac.ComputeFeasibleVals();
    spdlog::info("Material: lambda range [{}, {}], kappa range [{}, {}]",
                 ac.range_lam.x, ac.range_lam.y, ac.range_kap.x, ac.range_kap.y);

    // 4. Load seg_id
    std::vector<int> seg_id = loadSegId(segid_path);
    if ((int)seg_id.size() != nF_global) {
        spdlog::error("seg_id size ({}) != mesh face count ({})", seg_id.size(), nF_global);
        return 1;
    }

    // 5. Initialize global metrics arrays
    Eigen::VectorXd stretch_excess = Eigen::VectorXd::Zero(nF_global);
    Eigen::VectorXd bend_excess    = Eigen::VectorXd::Zero(nF_global);

    // 6. Determine number of patches
    int num_patches = *std::max_element(seg_id.begin(), seg_id.end()) + 1;
    spdlog::info("Processing {} patches", num_patches);

    // 7. Process each patch
    for (int pid = 0; pid < num_patches; ++pid) {
        PatchData patch = extractPatch(V, F, seg_id, pid);
        if (patch.F.rows() == 0) {
            spdlog::warn("Patch {} has 0 faces, skipping", pid);
            continue;
        }
        spdlog::info("Patch {}: {} faces, {} vertices", pid, patch.F.rows(), patch.V.rows());

        // 7b. Parameterization (material-constrained ASAP)
        Eigen::MatrixXi F_p = patch.F;
        int nF_orig = F_p.rows();

        std::vector<int> boundary = fillInHoles(patch.V, F_p);
        Eigen::MatrixXd P_p = tutteEmbedding(patch.V, F_p, boundary);

        LocalGlobalSolver solver(patch.V, F_p);
        solver.solve(P_p, 1.0 / ac.range_lam.y, 1.0 / ac.range_lam.x);
        centerAndRotate(patch.V, P_p);

        // Restore original face count (remove filled-in faces)
        F_p.conservativeResize(nF_orig, 3);

        // 7c. Second scaling: fit parameterization to platewidth
        double s2 = platewidth / (P_p.colwise().maxCoeff() - P_p.colwise().minCoeff()).maxCoeff();
        Eigen::MatrixXd V_scaled = patch.V * s2;
        P_p *= s2;

        // 7d. Compute morphing parameters
        ManifoldSurfaceMesh mesh_p(F_p);
        VertexPositionGeometry geom_p(mesh_p, V_scaled);
        geom_p.refreshQuantities();

        FaceData<Eigen::Matrix2d> MrInv_p = precomputeMrInv(mesh_p, P_p, F_p);

        int nV_p = V_scaled.rows();
        int nF_p = F_p.rows();

        Eigen::VectorXd lam_pv = Eigen::VectorXd::Zero(nV_p);
        Eigen::VectorXd lam_pf = Eigen::VectorXd::Zero(nF_p);
        Eigen::VectorXd kap_pv = Eigen::VectorXd::Zero(nV_p);
        Eigen::VectorXd kap_pf = Eigen::VectorXd::Zero(nF_p);

        std::vector<bool> bv_flags(nV_p, false);
        std::vector<bool> bf_flags(nF_p, false);
        std::vector<int>  b_ref(nF_p, 0);

        Morphmesh::ComputeMorphophing(geom_p, V_scaled, F_p,
            nV_p, nF_p,
            bv_flags, bf_flags, b_ref, MrInv_p,
            lam_pv, lam_pf, kap_pv, kap_pf);

        // 7e. Compute feasibility excess and map back to global
        for (int i = 0; i < nF_p; ++i) {
            double lam = lam_pf[i];
            double kap = kap_pf[i];
            double s_ex = std::max(0.0, lam - ac.range_lam.y)
                        + std::max(0.0, ac.range_lam.x - lam);
            double b_ex = std::max(0.0, std::abs(kap) - ac.range_kap.y);
            stretch_excess[patch.global_face_ids[i]] = s_ex;
            bend_excess[patch.global_face_ids[i]]    = b_ex;
        }

        spdlog::info("  Patch {} done: lambda [{:.4f}, {:.4f}], kappa [{:.4f}, {:.4f}]",
                     pid, lam_pf.minCoeff(), lam_pf.maxCoeff(),
                     kap_pf.minCoeff(), kap_pf.maxCoeff());
    }

    // 8. Output metrics.txt
    std::ofstream ofs(output_path);
    if (!ofs.is_open()) {
        spdlog::error("Cannot open output file: {}", output_path);
        return 1;
    }
    for (int f = 0; f < nF_global; ++f) {
        ofs << stretch_excess[f] << " " << bend_excess[f] << "\n";
    }
    ofs.close();

    spdlog::info("Wrote metrics to {} ({} faces)", output_path, nF_global);

    // Summary statistics
    int n_infeasible = 0;
    for (int f = 0; f < nF_global; ++f) {
        if (stretch_excess[f] > 0 || bend_excess[f] > 0)
            n_infeasible++;
    }
    spdlog::info("Infeasible faces: {}/{} ({:.1f}%)",
                 n_infeasible, nF_global, 100.0 * n_infeasible / nF_global);

    return 0;
}
