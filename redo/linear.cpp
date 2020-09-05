#include "linear.h"

const double LinearApprox::matrix[3][3] = {{9,-3,-3}, {-3,9,-3}, {-3,-3,9}};

LinearApprox::LinearApprox(Imagem& img, double step, double ds) : Approx(img, step, ds) {
}

LinearApprox::~LinearApprox() {
    for(int i = 0; i < numTri; i++) {
        delete[] coefficients[i];
        delete[] basisIntegral[i];
    }
    delete[] coefficients;
    delete[] basisIntegral;
}

ApproxType LinearApprox::getApproxType() {
    return APPROXTYPE;
}

void LinearApprox::reallocateSpace(int oldNumTri) {
    for(int i = 0; i < oldNumTri; i++) {
        delete[] coefficients[i];
        delete[] basisIntegral[i];
    }
    delete[] coefficients;
    delete[] basisIntegral;
    coefficients = new double *[numTri];
    basisIntegral = new double *[numTri];
    for(int i = 0; i < numTri; i++) {
        coefficients[i] = new double[3];
        basisIntegral[i] = new double[3];
    }
}

void LinearApprox::initialize(int pixelRate) {
    Approx::initialize(APPROXTYPE, pixelRate);
    coefficients = new double *[numTri];
    basisIntegral = new double *[numTri];
    for(int i = 0; i < numTri; i++) {
        coefficients[i] = new double[3];
        basisIntegral[i] = new double[3];
    }
    updateApprox();
}

void LinearApprox::initialize(vector<Point> &pts, vector<array<int, 3>> &faces) {
    Approx::initialize(APPROXTYPE, pts, faces);
    coefficients = new double *[numTri];
    basisIntegral = new double *[numTri];
    for(int i = 0; i < numTri; i++) {
        coefficients[i] = new double[3];
        basisIntegral[i] = new double[3];
    }
    updateApprox();
}

double LinearApprox::computeEnergy() {
    double totalEnergy = 0;
    for(int t = 0; t < numTri; t++) {
        array<int, 3> vertices = faces.at(t);
        totalEnergy += integrator.linearEnergyApprox(triArr + t, coefficients[t], ds);
    }
    return totalEnergy;
}

void LinearApprox::gradient(int t, int movingPt, double *gradX, double *gradY) {
    double area = triArr[t].getArea();
    double gradient[2] = {0,0};
    double dA[2] = {triArr[t].gradX(movingPt), triArr[t].gradY(movingPt)};

    double dL[2][3]; // gradient of f phi_j dA; first row in x direction, second in y
    for(int i = 0; i < 2; i++) {
        integrator.linearImageGradient(triArr + t, movingPt, (i == 0), ds, dL[i]); // area integral
        for(int j = 0; j < 3; j++) {
            dL[i][j] += integrator.lineIntEval(triArr + t, movingPt, (i == 0), ds/2, j); // boundary integral
        }
    }

   double dLA[2][3]; // gradient of d(L/A)
   for(int i = 0; i < 2; i++) {
       for(int j = 0; j < 3; j++) {
           dLA[i][j] = (area * dL[i][j] - basisIntegral[t][j] * dA[i]) / (area * area);
       }
   }

    double dc[2][3] = {{0,0,0},{0,0,0}}; // gradients of coefficients
    for(int j = 0; j < 3; j++) {
        for(int k = 0; k < 3; k++) {
            dc[0][j] += matrix[j][k] * dLA[0][k];
            dc[1][j] += matrix[j][k] * dLA[1][k];
        }
    }

    // update energy gradient
   for(int i = 0; i < 2; i++) {
       for(int j = 0; j < 3; j++) {
            gradient[i] -= (dc[i][j] * basisIntegral[t][j] + coefficients[t][j] * dL[i][j]);
       }
   }

    // in case of small area, only use log barrier gradient
    // to move away from small area
    for(int i = 0; i < 2; i++) {
        if(isnan(gradient[i])) {
            assert(area < 2 * AREA_THRESHOLD);
            gradient[i] = 0;
        }
    }

    // account for image boundary and log area barrier 
    regularizationGrad(t, movingPt, gradient[0], gradient[1]);

    if (gradX && gradY) {
		*gradX = gradient[0];
		*gradY = gradient[1];
	}
}

void LinearApprox::updateApprox() {
    for(int t = 0; t < numTri; t++) {
		// compute image phi_i dA and store it for reference on next iteration
        integrator.doubleIntEval(triArr + t, ds, basisIntegral[t]);
		double area = triArr[t].getArea();
        // extract coefficients
        for(int i = 0; i < 3; i++) {
            double coeff = 0;
            for(int j = 0; j < 3; j++) {
                coeff += matrix[i][j] * basisIntegral[t][j];
            }
            coefficients[t][i] = min(255.0, coeff / area); // prevent blowup
        }
	}
}

void LinearApprox::computeCoeffs(Triangle *tri, double *coeffs, ColorChannel channel) {
    double L[3]; // integral of f phi_i dA
    integrator.doubleIntEval(tri, ds, L, channel);
    for(int i = 0; i < 3; i++) {
        double coeff = 0;
        for(int j = 0; j < 3; j++) {
            coeff += matrix[i][j] * L[j];
        }
        coeffs[i] = min(255.0, coeff / tri->getArea());
    }
}

void LinearApprox::computeEdgeEnergies(vector<array<double, 3>> *edgeEnergies) {
    const bool salient = true; // use saliency when picking edges to subdivide
    for(auto ii = edgeBelonging.begin(); ii != edgeBelonging.end(); ii++) {
		array<int, 2> edge = ii->first;
		vector<int> triangles = ii->second; // triangles containing edge
		// compute current total energy over these triangles
		double curEnergy = 0;
		for(int t : triangles) {
			curEnergy += integrator.linearEnergyApprox(triArr+t, coefficients[t], ds, salient) + regularizationEnergy(triArr + t);
		}
		// find new point that may be added to mesh
		Point endpoint0 = points[edge[0]];
		Point endpoint1 = points[edge[1]];
		double midX = (endpoint0.getX() + endpoint1.getX()) / 2;
		double midY = (endpoint0.getY() + endpoint1.getY()) / 2; 
		Point midpoint(midX, midY);

		double newEnergy = 0;
		for(int t : triangles) {
			Point opposite;
			// get opposite vertex
			for(int v = 0; v < 3; v++) {
				// for accuracy, use raw indices rather than point location
				if(faces.at(t).at(v) != edge[0] && faces.at(t).at(v) != edge[1]) {
					opposite = points[faces.at(t).at(v)];
				}
			}
			// the two triangles formed by cutting this edge
			Triangle t1(&midpoint, &opposite, &endpoint0);
			Triangle t2(&midpoint, &opposite, &endpoint1);
			// equal area of both triangles
			double area = triArr[t].getArea() / 2;
			// get energy on subdivided triangles
            double coeffs1[3];
            double coeffs2[3];
            computeCoeffs(&t1, coeffs1);
            computeCoeffs(&t2, coeffs2);
			newEnergy += integrator.linearEnergyApprox(&t1, coeffs1, ds, salient) + integrator.linearEnergyApprox(&t2, coeffs2, ds, salient)
                + 2 * regularizationEnergy(&t1) + EDGE_SPLIT_MULTIPLIER / endpoint0.distance(endpoint1);
		}
		// change in energy due to subdivision
		edgeEnergies->push_back({(double) edge[0], (double) edge[1], newEnergy - curEnergy});
	}
}

vector<array<double, 3>> LinearApprox::getColors() {
    vector<array<double, 3>> colors;
    const int scale = 255;
    for(int t = 0; t < numTri; t++) {
        double area = triArr[t].getArea();
        // one vertex at a time and extract rgb separately
        double r[3];
        double g[3];
        double b[3];
        computeCoeffs(triArr + t, r, RED);
        computeCoeffs(triArr + t, g, GREEN);
        computeCoeffs(triArr + t, b, BLUE);
        for(int i = 0; i < 3; i++) {
            colors.push_back({r[i]/scale, g[i]/scale, b[i]/scale});
        }
    }
    return colors;
}