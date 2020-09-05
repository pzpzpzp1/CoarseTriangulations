#ifndef approx_h
#define approx_h

#include <map>
#include <vector>
#include <array>
#include <set>
#include "triangle.cuh"
#include "parallelInt.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "Imagem.h"

using namespace std;

/**
 * interface for triangular image approximation
 * To use, one of the initialize functions must be called;
 * thereafter the run functions runs the entire gradient flow
 * to convergence. Note that this does not support adaptive
 * re-meshing; that requires the subdivide function to be called
 * separately.
 */

class Approx {
	protected:
		static constexpr double MIN_STEP = 1e-07; // minimum acceptable step size for avoiding triangle inversions
		static constexpr double ABSOLUTE_MIN = 1e-24; // absolute minimum step size for ensuring energy decrease
		int maxX, maxY; // dimensions of image
		double stepSize; // size of gradient descent step
		double originalStep; // starting step size
		Pixel *pixArr; // pixel (x, y) is pixArr[x * maxY + y]
		Point *points; // store vertices of triangulation
		int numPoints;
		vector<array<int, 3>> faces; // hold triangle connections; MUST BE IN CCW ORDER
		Triangle *triArr; // store triangles of triangulation
		int numTri; // number of triangles
		map<Point*, double> gradX; // map points to gradient x values
		map<Point*, double> gradY; // map points to gradient y values
		double ds; // step size for integration approximation
		ParallelIntegrator integrator; // do all the integrations

		map<array<int, 2>, vector<int>> edgeBelonging; // map edge to indices of triangles containing edge
		// edge represented by sorted indices of points

		set<int> tinyTriangles; // triangles at risk of inverting
		bool zeroed = false; // whether image gradients of tiny triangles have already been zeroed
		bool areaThrottled = false; // whether total gradients of tiny triangles have been zeroed

        // initialize triangulation and integrator
		// initialize the triangulation on this approximation using a coarse grid,
		// sampling once every pixelRate pixels
		void initialize(ApproxType approxtype, int pixelRate);
		// initialize a constant approximation triangulation on img
		// with input triangulation points, faces
		void initialize(ApproxType approxtype, vector<Point> &points, vector<array<int, 3>> &triangleInd);

	public:
		// create an approximation instance on img with given stepsize and sampling rate
		Approx(Imagem& img, double step, double ds = 0.1);
		// deallocate all the shared space
		virtual ~Approx() = 0;
        virtual ApproxType getApproxType() = 0;

        // free and reassign space that is not handled in approx 
		// pass in old number of triangles for 2D arrays
        virtual void reallocateSpace(int oldNumTri) = 0;

		// set saliency map from a 1D saliency vector, where columns are contiguous
		void setSaliency(vector<double> saliency);
        // create starting approximation
        virtual void initialize(int pixelRate) = 0;
        virtual void initialize(vector<Point> &points, vector<array<int, 3>> &triangleInd) = 0;

		// compute approximation energy of triangulation at this point in time
		virtual double computeEnergy() = 0;
		// compute energy of tuning parameters, i.e. negative log area barrier
		double regularizationEnergy();
		// compute regularization energy over tri
		double regularizationEnergy(Triangle *tri);
		// add gradient of regularization energy of i th point of
		// triArr[t] to gradX, gradY; sets these to 0 appropriately
		// if point is on x or y boundaries of image
		void regularizationGrad(int t, int i, double &gradX, double &gradY);
		// compute gradient at this instant, updating gradX and gradY
		void computeGrad();
		// helper function for computeGrad;
		// store gradient values in gradX, gradY of energy over triangle triArr[t]
		// of the point in t with index movingPt
		virtual void gradient(int t, int movingPt, double *gradX, double *gradY) = 0;
		// move points according to gradient values and return true
		// if movement was successful, i.e. no triangle inverts under the process
		bool gradUpdate();
		// in the case that a triangle inverts, undo all the changes at this step
		// and halve the stepSize
		void undo();
		// update approximation value for each triangle
		virtual void updateApprox() = 0;

		// handle adaptive retriangulation to finer mesh

		/* greedily subdivide the top n edges */
		void subdivide(int n);
		// helper for subdivide
		// compute energy change associated with subdividing each edge at its midpoint
		// {i, j, e} in vector means edge (i, j) has energy e
		virtual void computeEdgeEnergies(vector<array<double, 3>> *edgeEnergies) = 0;
		// helper for subdivide
		// update subdivided mesh
		void updateMesh(vector<Point> *newPoints, vector<array<int, 3>> *newFaces, set<int> *discardedFaces);

		/* return data for image to be displayed */
		// get stepsize
		double getStep();
		// get points of triangulation
		vector<Point> getVertices();
		// get edges of triangulation
		vector<array<int, 3>> getFaces();
        // return colors in a 1D array; note for linear approximation
        // the colors will be ordered triangle 0 vertex 0, triangle 0 vertex 1, etc.
        virtual vector<array<double, 3>> getColors() = 0;
		// return set of indices of triangles at risk of inverting
		set<int> getTinyTriangles();

		// for triangulations where each triangle is rendered separately, compute bounding box
		// for display purposes
		vector<Point> boundingBox();
		vector<array<int, 3>> boundingFaces();

		// run one full step of the procedure while tracking energy values;
		// approxErr is new approximation energy after this step
		// return stepsize used in this step
		// stringent determines whether any energy increase is tolerated,
		// defaults true (no tolerance)
		double step(double &prevEnergy, double &newEnergy, double &approxErr, bool stringent = true);
		// run the entire procedure for either maxIter iterations or
		// until change in energy is at most eps
		void run(int maxIter = 100, double eps = 0.001);
};

#endif