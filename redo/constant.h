#ifndef constant_h
#define constant_h

#include <assert.h>
#include <map>
#include <vector>
#include <array>
#include <set>
#include "approx.h"
#include "Imagem.h"

using namespace std;

/**
 * compute a piecewise constant coarse triangular approximation of an image
 */

class Approx;

class ConstantApprox : public Approx {
	private:
		static const ApproxType APPROXTYPE = constant;
		double *imageInt; // hold integrals of image dA over each triangle
		double *grays; // grays[i] is the luminance of triangle triArr[i]

	public:
		// create an approximation instance on img with given stepsize and sampling rate
		ConstantApprox(Imagem& img, double step, double ds = 10);
		// deallocate all the shared space
		~ConstantApprox();
		ApproxType getApproxType();

		void reallocateSpace(int oldNumTri);

		void initialize(int pixelRate);
		void initialize(vector<Point> &points, vector<array<int,3>> &faces);

		void updateApprox();
		double computeEnergy();

		// store gradient values in gradX, gradY of energy over triangle triArr[t]
		// of the point in t with index movingPt
		void gradient(int t, int movingPt, double *gradX, double *gradY);

		void computeEdgeEnergies(vector<array<double, 3>> *edgeEnergies);

		vector<array<double, 3>> getColors();

		/*
		void registerMesh(bool first = false);
		void updateMesh();
		*/
};

#endif
