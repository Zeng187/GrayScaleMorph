/// Verify: test simulationFunction's bending energy on a known shape.
///
/// Given a flat param mesh and a 3D mesh (same topology), constructs
/// simulationFunction with a SWEEP of target kappa values, evaluates
/// the energy at the given 3D shape (no Newton solve), and reports
/// which kappa minimises the bending energy.
///
/// This directly tests whether simulationFunction "sees" the correct
/// curvature on the given mesh.
///
/// Usage:
///   ./Verify <param.obj> <mesh.obj>

#include <Eigen/Core>

#include <igl/loop.h>
#include <igl/readOBJ.h>

#include <spdlog/spdlog.h>

#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include <TinyAD/Support/GeometryCentral.hh>

#include <cstdlib>
#include <iostream>
#include <string>

#include "boundary_utils.h"
#include "functions.h"
#include "morph_functions.hpp"
#include "morphmesh.hpp"

using namespace geometrycentral;
using namespace geometrycentral::surface;

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <param.obj> <mesh.obj>\n";
        return 1;
    }

    try {
        const std::string paramPath = argv[1];
        const std::string meshPath  = argv[2];

        // -- Load param mesh (2D, z=0) ----------------------------------------
        Eigen::MatrixXd Vp;
        Eigen::MatrixXi Fp;
        if (!igl::readOBJ(paramPath, Vp, Fp) || Vp.rows() == 0)
            throw std::runtime_error("Could not read param mesh: " + paramPath);

        // -- Load 3D mesh -----------------------------------------------------
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        if (!igl::readOBJ(meshPath, V, F) || V.rows() == 0)
            throw std::runtime_error("Could not read 3D mesh: " + meshPath);

        // -- Loop-subdivide param to match 3D mesh if needed ------------------
        while (Fp.rows() < F.rows()) {
            Eigen::MatrixXd Vp3 = Eigen::MatrixXd::Zero(Vp.rows(), 3);
            Vp3.leftCols(Vp.cols()) = Vp;
            Eigen::MatrixXd Vp3_sub;
            Eigen::MatrixXi Fp_sub;
            igl::loop(Vp3, Fp, Vp3_sub, Fp_sub);
            Vp = Vp3_sub.leftCols(2);  // keep only xy for 2D param
            Fp = Fp_sub;
            spdlog::info("Loop-subdivided param -> {}V, {}F", Vp.rows(), Fp.rows());
        }

        Eigen::MatrixXd P = Vp.leftCols(2);  // Nx2

        if (V.rows() != P.rows() || F.rows() != Fp.rows())
            throw std::runtime_error(
                "Vertex/face count mismatch after subdivision: param "
                + std::to_string(P.rows()) + "V/" + std::to_string(Fp.rows())
                + "F vs mesh " + std::to_string(V.rows()) + "V/"
                + std::to_string(F.rows()) + "F");

        const int nV = static_cast<int>(V.rows());
        const int nF = static_cast<int>(F.rows());
        spdlog::info("Loaded: {} vertices, {} faces.", nV, nF);

        // -- Build geometry + MrInv from flat param ---------------------------
        Eigen::MatrixXd V_flat = Eigen::MatrixXd::Zero(nV, 3);
        V_flat.col(0) = P.col(0);
        V_flat.col(1) = P.col(1);

        ManifoldSurfaceMesh    mesh(F);
        VertexPositionGeometry geometry(mesh, V);
        geometry.refreshQuantities();

        FaceData<Eigen::Matrix2d> MrInv = precomputeMrInv(mesh, P, F);
        std::vector<bool> is_boundary;
        std::vector<int> ref_faces = buildRefFaces(mesh, is_boundary);

        // -- Flatten x vector from 3D mesh (the "deformed" state) -------------
        Eigen::VectorXd x = Eigen::VectorXd::Zero(3 * nV);
        for (int i = 0; i < nV; ++i) {
            x[3 * i + 0] = V(i, 0);
            x[3 * i + 1] = V(i, 1);
            x[3 * i + 2] = V(i, 2);
        }
        // Also set up via TinyAD's interface for correct variable mapping
        geometry.requireVertexIndices();

        // -- Set lambda = 1.0 (isometric bending, no stretching) --------------
        FaceData<double> lambda_pf(mesh, 1.0);

        // -- Sweep kappa and evaluate total energy ----------------------------
        constexpr double E_mod = 1.0;
        constexpr double nu    = 0.5;
        constexpr double h     = 1.0;
        constexpr double w_s   = 1.0;
        constexpr double w_b   = 1.0;

        spdlog::info("Sweeping kappa to find energy minimum...");
        spdlog::info("{:>10s} {:>15s} {:>15s} {:>15s}",
                     "kappa", "E_total", "E_stretch", "E_bend");

        double best_kap = 0;
        double best_energy = 1e30;

        for (int step = -20; step <= 120; ++step) {
            double kap_test = step * 0.001;  // sweep from -0.02 to 0.12

            FaceData<double> kappa_pf(mesh, kap_test);

            auto simFunc = simulationFunction(
                geometry, MrInv, lambda_pf, kappa_pf,
                E_mod, nu, h, w_s, w_b, ref_faces);

            // Evaluate energy at the 3D shape (no solve)
            auto x_from = simFunc.x_from_data(
                [&](Vertex v) { return V.row(geometry.vertexIndices[v]); });
            double energy = simFunc.eval(x_from);

            // Also evaluate with w_b=0 to get stretching only
            FaceData<double> kappa_zero(mesh, 0.0);
            auto simFunc_s = simulationFunction(
                geometry, MrInv, lambda_pf, kappa_zero,
                E_mod, nu, h, w_s, 0.0, ref_faces);  // w_b=0
            double e_stretch = simFunc_s.eval(x_from);
            double e_bend = energy - e_stretch;

            spdlog::info("{:10.4f} {:15.8f} {:15.8f} {:15.8f}",
                         kap_test, energy, e_stretch, e_bend);

            if (energy < best_energy) {
                best_energy = energy;
                best_kap = kap_test;
            }
        }

        spdlog::info("Energy-minimising kappa = {:.4f} (total E = {:.8f})",
                     best_kap, best_energy);
        spdlog::info("For cylinder R=10: expected H = 0.05");

    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        return 1;
    }
    return 0;
}
