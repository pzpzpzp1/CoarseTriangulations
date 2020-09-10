#ifndef simulator_h
#define simulator_h

#include <iostream>
//#include "MatlabEngine.hpp"
//#include "MatlabDataArray.hpp"
#include "constant.h"
#include "linear.h"
#include "Imagem.h"

/**
 * run gradient flow and subdivisions on a triangular mesh approximation
 * This is the wrapper class for triangular approximations and acts as the
 * backend interface with the user.
 */

//using namespace matlab::engine;

// rendering functions that are currently implemented in main file
// because I can't figure out how to make cmake cooperate with polyscope

// render mesh with colors
// first indicates whether this is the first registration of the mesh
void registerMesh(Approx *approx);
void updateMesh(Approx *approx);
// show edges if show, hide edges otherwise; toggles show
void showEdges(Approx *approx, bool &show);
// highlight triangle t in red if on
void highlight(ApproxType approx, int t, bool on = true);

class Simulator {
    private:
        static constexpr double DENSITY_DEFAULT = 0.05; // experimentally a decent starting density
        
        static constexpr double STARTING_STEP = 0.00625; // starting step size

        //std::unique_ptr<MATLABEngine> matlabPtr;
        //matlab::data::ArrayFactory factory;

        Approx *approx; // actual approximation instance

        // initialization values
        double density; // density input for TRIM
        int dx; // take one sample every dx pixels (uniform)

        // for rendering
        double prevEnergy = 0;
        double approxErr = 0;
        double newEnergy = 0;
        int iterCount = 0;
        int totalIters = 0; // total iterations
        vector<double> elapsedTimeVec; // hold cumulative step size
        vector<double> energyVec; // hold energy per iteration
        vector<double> errorVec; // hold approximation error per iteration
        double totalStep = 0; // running total step
        // in the linear case, track which triangles are currently highlighted
        // TODO: support constant case as well
        set<int> highlightedTriangles; 

        // for controlling convergence 
        int numSmallChanges = 0;
        static const int maxSmallChanges = 3;

    public:
        // create an approximation instance from an image path
        Simulator(string imgPath, Imagem& img, ApproxType approxtype, int dxIn);
        ~Simulator();

        // display the starting triangulation
        void initialize();
        // run one step of gradient flow, incrementing numSmallChanges
        // if a change within a multiplicative factor of eps is detected
        void step(double eps);
        // run one step of gradient flow with stopping conditions;
        // return true if a step was made
        bool step(int maxIter = 100, double eps = 0.001);
        // run gradient flow to convergence
        void flow(int maxIter = 100, double eps = 0.001);

        // handle retriangulation of num edges
        void retriangulate(int num);

        // show all edges if display, otherwise hide edges;
        // toggle display
        void revealEdges(bool &display);
        
        // handle output graphs (energy and elapsed time)
        void cleanup();
};

#endif