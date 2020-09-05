#include "constant.h"

const double TOLERANCE = 1e-10;

ConstantApprox::ConstantApprox(Imagem& img, double step, double ds) : Approx(img, step, ds) {
}

ApproxType ConstantApprox::getApproxType() {
	return APPROXTYPE;
}

void ConstantApprox::reallocateSpace(int oldNumTri) {
	delete[] imageInt;
	delete[] grays;
	imageInt = new double[numTri];
	grays = new double[numTri];
}

void ConstantApprox::initialize(vector<Point> &pts, vector<array<int, 3>> &inds) {
	Approx::initialize(APPROXTYPE, pts, inds); // call parent method
	imageInt = new double[numTri];
	grays = new double[numTri];

	// create an initial approximation based on this triangulation
	updateApprox();
}

void ConstantApprox::initialize(int pixelRate) {
	Approx::initialize(APPROXTYPE, pixelRate);
	imageInt = new double[numTri];
	grays = new double[numTri];
	// create an initial approximation based on this triangulation
	updateApprox();
}

ConstantApprox::~ConstantApprox() {
	delete[] grays;
	delete[] imageInt;
}

double ConstantApprox::computeEnergy() {
	double totalEnergy = 0;
    for(int t = 0; t < numTri; t++) {
        totalEnergy += integrator.constantEnergyEval(triArr + t, grays[t], ds);
    }
    return totalEnergy;
}

void ConstantApprox::gradient(int t, int movingPt, double *gradX, double *gradY) {
	// to save time, only compute integrals if triangle is non-degenerate;
	// degenerate triangle has 0 energy and is locally optimal, set gradient to 0
	double area = triArr[t].getArea();
	double gradient[2] = {0, 0};
	double imageIntegral = imageInt[t];
	if (area > TOLERANCE) {
		double dA[2] = {triArr[t].gradX(movingPt), triArr[t].gradY(movingPt)};
		double boundaryChange[2];
		// compute gradient in x and y direction
		for(int i = 0; i < 2; i++) {
			// sample more frequently because both time and space allow (or don't)
			boundaryChange[i] = integrator.lineIntEval(triArr+t, movingPt, (i == 0), ds);
		}
		for(int j = 0; j < 2; j++) {
			gradient[j] = (2 * area * imageIntegral * boundaryChange[j]
				- imageIntegral * imageIntegral * dA[j]) / (-area * area);
		}
	}
	regularizationGrad(t, movingPt, gradient[0], gradient[1]);
	// check for null pointers
	if (gradX && gradY) {
		*gradX = gradient[0];
		*gradY = gradient[1];
	}
}

void ConstantApprox::updateApprox() {
	for(int t = 0; t < numTri; t++) {
		// compute image dA and store it for reference on next iteration
		double val;
		integrator.doubleIntEval(triArr + t, ds, &val);
		imageInt[t] = val;
		double area = triArr[t].getArea();
		// take average value
		double approxVal = val / area;
		// handle degeneracy
		if (isnan(approxVal)) {
			assert(area < TOLERANCE);
			approxVal = 255; // TODO: something better than this
		}
		grays[t] = min(255.0, approxVal); // prevent blowup in case of poor approximation
	}
}

void ConstantApprox::computeEdgeEnergies(vector<array<double, 3>> *edgeEnergies) {
	const bool salient = true; // want saliency when picking edges to subdivide
	for(auto ii = edgeBelonging.begin(); ii != edgeBelonging.end(); ii++) {
		array<int, 2> edge = ii->first;
		vector<int> triangles = ii->second; // triangles containing edge
		// compute current total energy over these triangles
		double curEnergy = 0;
		for(int t : triangles) {
			curEnergy += integrator.constantEnergyEval(triArr+t, grays[t], ds, salient) + regularizationEnergy(triArr + t);
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
			double color1, color2;
			integrator.doubleIntEval(&t1, ds, &color1);
			integrator.doubleIntEval(&t2, ds, &color2);
			color1 /= area;
			color2 /= area;
			newEnergy += integrator.constantEnergyEval(&t1, color1, ds, salient) + integrator.constantEnergyEval(&t2, color2, ds, salient)
				+ 2 * regularizationEnergy(&t1) // regularization energies on t1, t2 are the same
				+ EDGE_SPLIT_MULTIPLIER / endpoint0.distance(endpoint1); // penalize short edges
		}
		// change in energy due to subdivision
		edgeEnergies->push_back({(double) edge[0], (double) edge[1], newEnergy - curEnergy});
	}
}

vector<array<double,3>> ConstantApprox::getColors() {
	vector<array<double, 3>> fullColors;
	for(int t = 0; t < numTri; t++) {
		// scale to fit polyscope colors 
		int scale = 255;
		double area = triArr[t].getArea();
		double r, g, b;
		integrator.doubleIntEval(triArr+t, ds, &r, RED);
		integrator.doubleIntEval(triArr+t, ds, &g, GREEN);
		integrator.doubleIntEval(triArr+t, ds, &b, BLUE);
		r /= (scale * area);
		g /= (scale * area);
		b /= (scale * area);
		fullColors.push_back({r, g, b});
	}
	return fullColors;
}

/*
void ConstantApprox::registerMesh(bool first) {
	auto triangulation = polyscope::registerSurfaceMesh2D("Triangulation", getVertices(), getFaces());
    auto colors = triangulation->addFaceColorQuantity("approximate colors", getColors());
	if(first) {
		// allow colors by default
    	colors->setEnabled(true);
    	// set material to flat to get more accurate rgb values
    	triangulation->setMaterial("flat");
	}
}
*/

/*
void ConstantApprox::updateMesh() {
	auto triangulation = polyscope::getSurfaceMesh("Triangulation");
    triangulation->updateVertexPositions2D(getVertices());
    triangulation->addFaceColorQuantity("approximate colors", getColors());
}
*/