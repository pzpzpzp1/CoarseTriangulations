#ifndef linear_h
#define linear_h

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

class LinearApprox : public Approx {
	private:
		static const ApproxType APPROXTYPE = linear;
        static const double matrix[3][3]; // matrix multiplier to obtain coefficients

        double **coefficients; // numTri x 3 array where the ijth entry is the coeff of phi_j on T_i
        double **basisIntegral; // numTri x 3, ijth entry is integral f phi_j dA on T_i

	public:
		// create an approximation instance on img with given stepsize and sampling rate
		LinearApprox(Imagem& img, double step, double ds = 0.1);
		// deallocate all the shared space
		~LinearApprox();
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

		// compute the coefficients on triangle tri and store in c,
		// aligned with coefficients of tri
		void computeCoeffs(Triangle *tri, double *c, ColorChannel channel = GRAY);
};

#endif