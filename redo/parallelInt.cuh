#ifndef parallel_int_cuh
#define parallel_int_cuh
#include <stdio.h>
#include <assert.h>
#include "pixel.cuh"
#include "triangle.cuh"
#include "cuda_runtime.h"

using namespace std;

// define different supported approximation types
// with values the number of FEM basis elements
enum ApproxType{constant = 1, linear = 3, quadratic = 6};

// multiplication factor on log area barrier
static const double LOG_AREA_MULTIPLIER = 100;
// multiplication factor on edge splitting in subdivision step
static const double EDGE_SPLIT_MULTIPLIER = 100000;
// do not tolerate triangles with area less than this
static const double AREA_THRESHOLD = 0.1;

// parallelize integral computation

class ParallelIntegrator {
    private:
        // thread setup
        static const int threadsX = 32;
        static const int threadsY = 16;
        static const int threads1D = 1024; // NOTE: changing this will require changes in sumBlock
        dim3 threads2D;

        bool initialized = false; // for sake of memory freeing, determine whether integrator is initialized

        // large array to hold parallel computations per basis element
        double **arr;
        // large array to hold partial sum computations
        double *helper;
        ApproxType approx; // type of approximation to do
        Pixel *pixArr; // reference to the image being approximated
        Point *curTri; // hold vertices of current working triangle
        int maxX, maxY; // size of image
        // true if computations are exact rather than approximate
        bool computeExact;

        // sum the first size values of arr[i]
        double sumArray(int size, int i = 0);

    public:
        ParallelIntegrator();
        // initialize parallel integrator, where pixel array pix is already initialized in shared memory
        // space indicates the amount of computation space needed
        // default to using approximate integrals
        // return true if successful
        bool initialize(Pixel *pix, int xMax, int yMax, ApproxType a, long long space, bool exact = false);
        // free allocated space
        ~ParallelIntegrator();

        // actual integrals

        // for energy, it makes sense to separate approximation types into different functions
        // because the inputs will be greatly different (one array for each coefficient)

        // compute energy integral over tri, possibly accounting for saliency
        double constantEnergyEval(Triangle *tri, double color, double ds, bool salient = false);
        // compute energy by exact integral
        double constantEnergyExact(Triangle *tri, double color, bool salient);
        // approximate energy using barycentric sampling
        double constantEnergyApprox(Triangle *tri, double color, double ds, bool salient);

        // compute integral of (f-g)^2 dA over triangle abc where
        // approximation is coeff[0] * a + coeff[1] * b + coeff[2] * c
        double linearEnergyApprox(Triangle *tri, double *coeffs, double ds, bool salient = false);

        // compute exact line integral (v dot n) * f phi ds over triangle tri
        // where FEM basis phi is dependent on approx and corresponds to basisInd of tri
        // consider when point with index pt in triArr[t] is moving at velocity (1,0) if isX and (0,1) if !isX
        double lineIntEval(Triangle *tri, int pt, bool isX, double ds, int basisInd = 0);
        // compute by exact integral
        double lineIntExact(Triangle *tri, int pt, bool isX);
        // approximate line integral using one sample every ds length
        double lineIntApprox(Triangle *tri, int pt, bool isX, double ds, int basisInd);

        // compute double integral of f phi dA over triArr[t], storing ith result in result[i]
        void doubleIntEval(Triangle *tri, double ds, double *result, ColorChannel channel = GRAY);
        // compute by exact integral
        double doubleIntExact(Triangle *tri, ColorChannel channel);
        // approximate by grid with side length ds
        void doubleIntApprox(Triangle *tri, double ds, double *result, ColorChannel channel);

        // compute integral f d phi_j dA over tri when pt is moving
        // store result in result[j]
        void linearImageGradient(Triangle *tri, int pt, bool isX, double ds, double *result);
};

#endif
