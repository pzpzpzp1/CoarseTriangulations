#include "approx.h"

static const int maxSmallChanges = 3;

Approx::Approx(Imagem & img, double step, double ds_) : originalStep(step), stepSize(step), ds(ds_) {
	// create pixel array representation
	maxX = img.width;
	maxY = img.height;
	// allocate shared space for pixel array
	cudaMallocManaged(&pixArr, maxX * maxY * sizeof(Pixel));
	for(int x = 0; x < maxX; x++) {
		for(int y = 0; y < maxY; y++) {
			int ind = x * maxY + y; // 1D pixel index
			
			int rgb[3];
			for(int i = 0; i < 3; i++) {
				rgb[i] = img.get(i, x, y);
			}
			// int r = img.get(0, x, y);
			pixArr[ind] = Pixel(x, y, rgb[0], rgb[1], rgb[2]);
		}
	}
}

Approx::~Approx() {
    cudaFree(pixArr);
    cudaFree(points);
    cudaFree(triArr);
}

void Approx::setSaliency(vector<double> saliency) {
	// check size requirements
	assert(saliency.size() == maxX * maxY);
	for(int i = 0; i < maxX * maxY; i++) {
		pixArr[i].setSaliency(saliency.at(i));
	}
}

void Approx::initialize(ApproxType approxtype, int pixelRate) {
	// create points
	int numX = ceil(((double) maxX) / pixelRate) + 1; // number of samples in x direction
	int numY = ceil(((double) maxY) / pixelRate) + 1;
	double dx = ((double) maxX) / (numX - 1); // step size in x direction, remembering to get both endpoints
	double dy = ((double) maxY) / (numY - 1);

	// create shared space for points
	numPoints = numX * numY;
	cudaMallocManaged(&points, numPoints * sizeof(Point));

	for(int i = 0; i < numX; i++) {
		bool isBoundX = (i == 0) || (i == numX - 1); // whether point is on vertical boundary
		for(int j = 0; j < numY; j++) {
			bool isBoundY = (j == 0) || (j == numY - 1);
			int index1D = i * numY + j;
			// shift by (-0.5, -0.5) to align to edge of image (lattice points at pixel centers)
			points[index1D] = Point(i * dx - 0.5, j * dy - 0.5, isBoundX, isBoundY);
		}
	}
	cout << "starting grid: " << numX << "x" << numY << endl;

	// create triangles
	numTri = 2 * (numX - 1) * (numY - 1);
	cudaMallocManaged(&triArr, numTri * sizeof(Triangle));

	int triInd = 0; // index the triangles
	for(int i = 0; i < numX; i++) {
		for(int j = 0; j < numY; j++) {
			int index1D = i * numY + j;
			// randomly triangulate the square with min x,y corner at this point
			if(i < numX - 1 && j < numY - 1) {
				Point *pt = points + index1D; // easier reference to current point
				// make sure face indices are ccw
				if(rand() % 2 == 0) {
					triArr[triInd] = Triangle(pt, pt + numY, pt + numY + 1);
					faces.push_back({index1D, index1D + numY, index1D + numY + 1});
					triArr[triInd+1] = Triangle(pt, pt + numY + 1, pt + 1);
					faces.push_back({index1D, index1D + numY + 1, index1D + 1});
				} else {
					triArr[triInd] = Triangle(pt, pt + numY, pt + 1);
					faces.push_back({index1D, index1D + numY, index1D + 1});
					triArr[triInd+1] = Triangle(pt + numY, pt + 1, pt + numY + 1);
					faces.push_back({index1D + numY, index1D + numY + 1, index1D + 1});
				}
				triInd += 2;
			}
		}
	}
	assert(triInd == numTri);

	// initialize edge dictionary from faces
	for(int i = 0; i < numTri; i++) {
		for(int j = 0; j < 3; j++) {
			array<int, 2> edge = {faces.at(i)[j], faces.at(i)[((j+1)%3)]};
			// ensure edges are ordered
			if(edge[0] > edge[1]) {
				edge[0] = edge[1];
				edge[1] = faces.at(i)[j];
			}
			if(edgeBelonging.find(edge) == edgeBelonging.end()) { // does not exist
				edgeBelonging[edge] = vector<int>();
			}
			edgeBelonging[edge].push_back(i);
		}		
	}

	// initialize integrator
	double maxLength = 2 * max(dx, dy); // generously round up maximum triangle side length

	// find space needed for results, one slot per gpu worker
	long long maxDivisions = (int) (maxLength/ds + 1); // max num samples per side, rounded up
	// maximum possible number of samples per triangle is loosely upper bounded by 2 * maxDivisions^2
	// assumming edge lengths are bounded above by maxDivisions * 2
	long long resultSlots = max(2 * maxDivisions * maxDivisions, (long long) maxX * maxY); // at least num pixels
	if(!integrator.initialize(pixArr, maxX, maxY, approxtype, resultSlots)) {
		exit(EXIT_FAILURE);
	}
}

void Approx::initialize(ApproxType approxtype, vector<Point> &pts, vector<array<int, 3>> &inds) {
	// load in points of triangulation
	numPoints = pts.size();
	// allocate shared space for points
	cudaMallocManaged(&points, numPoints * sizeof(Point));
	// copy everything in TODO: make this more efficient (get directly from source)
	for(int i = 0; i < numPoints; i++) {
		points[i] = pts.at(i);
	}

	// now load in all the triangles
	numTri = inds.size();
	// allocate shared space for triangles
	cudaMallocManaged(&triArr, numTri * sizeof(Triangle));

	double maxLength = 0; // get maximum side length of a triangle for space allocation
	faces = inds;
	for(int i = 0; i < numTri; i++) {
		array<int, 3> t = inds.at(i); // vertex indices for this triangle
		// constructor takes point addresses
		triArr[i] = Triangle(points + t.at(0), points + t.at(1), points + t.at(2));
		// ensure faces holds vertices in ccw order
		if(Triangle::getSignedArea(points + t.at(0), points + t.at(1), points + t.at(2)) < 0) {
			// flip second and third indices to match Triangle constructor
			faces.at(i).at(1) = t.at(2);
			faces.at(i).at(2) = t.at(1);
		}
		maxLength = max(maxLength, triArr[i].maxLength());
	}

	// initialize edge dictionary from faces
	for(int i = 0; i < numTri; i++) {
		for(int j = 0; j < 3; j++) {
			array<int, 2> edge = {faces.at(i)[j], faces.at(i)[((j+1)%3)]};
			// ensure edges are ordered
			if(edge[0] > edge[1]) {
				edge[0] = edge[1];
				edge[1] = faces.at(i)[j];
			}
			if(edgeBelonging.find(edge) == edgeBelonging.end()) { // does not exist
				edgeBelonging[edge] = vector<int>();
			}
			edgeBelonging[edge].push_back(i);
		}		
	}

	// initialize integrator

	// find space needed for results, one slot per gpu worker
	long long maxDivisions = (int) (maxLength/ds + 1); // max num samples per side, rounded up
	// maximum possible number of samples per triangle is loosely upper bounded by 2 * maxDivisions^2
	// assumming edge lengths are bounded above by maxDivisions * 2
	long long resultSlots = max(2 * maxDivisions * maxDivisions, (long long) maxX * maxY); // at least num pixels
	if(!integrator.initialize(pixArr, maxX, maxY, approxtype, resultSlots)) {
		exit(EXIT_FAILURE);
	}
}

double Approx::regularizationEnergy() {
	double energy = 0;
	for(int t = 0; t < numTri; t++) {
		energy -= LOG_AREA_MULTIPLIER * log(max(0.0, triArr[t].getArea() - AREA_THRESHOLD));
	}
	return energy;
}

double Approx::regularizationEnergy(Triangle *tri) {
	return -LOG_AREA_MULTIPLIER * log(max(0.0, tri->getArea() - AREA_THRESHOLD));
}

void Approx::regularizationGrad(int t, int i, double &gradX, double &gradY) {
	double area = triArr[t].getArea();
	double dA[2] = {triArr[t].gradX(i), triArr[t].gradY(i)};
	gradX -= LOG_AREA_MULTIPLIER * dA[0] / (area - AREA_THRESHOLD);
	gradY -= LOG_AREA_MULTIPLIER * dA[1] / (area - AREA_THRESHOLD);
	if(points[faces.at(t).at(i)].isBorderX()) gradX = 0;
	if(points[faces.at(t).at(i)].isBorderY()) gradY = 0;
}

void Approx::computeGrad() {
	// clear gradients from last iteration
	for(int i = 0; i < numPoints; i++) {
		gradX[points + i] = 0;
		gradY[points + i] = 0;
	}
	for(int i = 0; i < numTri; i++) {
		for(int j = 0; j < 3; j++) {
			double changeX, changeY;
			gradient(i, j, &changeX, &changeY);
			gradX[triArr[i].vertices[j]] += changeX;
			gradY[triArr[i].vertices[j]] += changeY;
		}
	}
}

bool Approx::gradUpdate() {
    // gradient descent update for each point
	for(int i = 0; i < numPoints; i++) {
		points[i].move(-stepSize * gradX.at(points+i), -stepSize * gradY.at(points+i));
	}
	// check validity of result
	bool inverted = false;
	for(int t = 0; t < numTri; t++) {
		if(triArr[t].getSignedArea() < AREA_THRESHOLD) {
			if(!zeroed || !areaThrottled) {
				tinyTriangles.insert(t);
			}
			inverted = true;
		}
	}
	return (!inverted);
}

void Approx::undo() {
    for(int i = 0; i < numPoints; i++) {
		points[i].move(stepSize * gradX.at(points+i), stepSize * gradY.at(points+i));
	}
	stepSize /= 2;
	if(!zeroed) { // still searching for small triangles
		if(stepSize < MIN_STEP) { // freeze these triangles
			for(int t : tinyTriangles) {
				for(int p : faces.at(t)) {
					gradX[points + p] = 0;
					gradY[points + p] = 0;
				}
			}
			// try to increase the area of these triangles
			// using log barrier gradient
			for(int t : tinyTriangles) {
				double area = triArr[t].getArea();
				for(int i = 0; i < 3; i++) {
					// set gradX[t[i]], gradY[t[i]] 
					regularizationGrad(t, i, gradX[points + faces.at(t).at(i)], gradY[points + faces.at(t).at(i)]);
				}
			}
			stepSize = originalStep;
			zeroed = true; // no need to check for small triangles anymore
		} else {
			tinyTriangles.clear();
		}
	} else if(!areaThrottled) { // small triangles have been determined
		if(stepSize < MIN_STEP) {
			// now even the log gradients are too large, so zero them out
			for(int t : tinyTriangles) {
				for(int p : faces.at(t)) {
					gradX[points + p] = 0;
					gradY[points + p] = 0;
				}
			}
			stepSize = originalStep;
			areaThrottled = true;
		} else {
			tinyTriangles.clear();
		}
	}
}

double Approx::step(double &prevEnergy, double &newEnergy, double &approxErr, bool stringent) {
	// reset status of tiny triangles
	tinyTriangles.clear();
	zeroed = false;
	areaThrottled = false;
	double usedStep;
	computeGrad();
    while(!gradUpdate()) {
        undo(); // keep halving stepSize until no triangle is inverted
	}
	updateApprox();
    prevEnergy = newEnergy;
	approxErr = computeEnergy();
	newEnergy = approxErr + regularizationEnergy();
    // TODO: tune this
	if(newEnergy > prevEnergy) { // overshot optimum?
		do {
        	do {
            	undo();
        	} while (!gradUpdate());
        	updateApprox();
			approxErr = computeEnergy();
			newEnergy = approxErr + regularizationEnergy();
			// prevent infinite loop
			if(stepSize < ABSOLUTE_MIN) {
				break;
			}
		} while(stringent && newEnergy > prevEnergy);
		usedStep = stepSize;
    } else {
		usedStep = stepSize;
		stepSize *= 2; // prevent complete vanishing to zero
	}
    cout << "new energy: " << newEnergy << endl;
	cout << "step size: " << usedStep << endl;
	return usedStep;
}

void Approx::run(int maxIter, double eps) {
	// track change in energy for stopping point
	double approxErr = computeEnergy();
	double newEnergy = approxErr + regularizationEnergy();
	// initialize to something higher than newEnergy
	double prevEnergy = newEnergy + 100 * eps;
	int iterCount = 0;
	int numSmallChanges = 0;
	while(iterCount < maxIter && numSmallChanges < maxSmallChanges) {
		cout << "iteration " << iterCount << endl;
		step(prevEnergy, newEnergy, approxErr);
		if(abs(prevEnergy - newEnergy) > eps * abs(prevEnergy)) {
        	numSmallChanges = 0;
    	} else {
        	numSmallChanges++;
    	}
		iterCount++;
	}
}

void Approx::subdivide(int n) {
	vector<array<double, 3>> edgeEnergies;
	computeEdgeEnergies(&edgeEnergies);
	// sort by energies
	sort(edgeEnergies.begin(), edgeEnergies.end(), [](const array<double, 3> a, const array<double, 3> b) {
		return a[2] < b[2];
	});

	// proceed through edgeEnergies and extract up to n edges to subdivide
	set<int> trianglesToRemove; // hold indices of triangles in triArr
	vector<Point> newPoints; // points to append
	vector<array<int, 3>> newTriangles; // new faces to add

	int numDivided = 0; // number of edges already divided
	int curIndex = 0; // current edge being considered
	while(numDivided < n && curIndex < edgeEnergies.size()) {
		array<int, 2> edge = {(int) edgeEnergies.at(curIndex)[0], (int) edgeEnergies.at(curIndex)[1]};
		vector<int> incidentFaces = edgeBelonging.at(edge);
		// for clean subdivision, don't cut the same face twice
		bool alreadyDivided = false;
		for(int t : incidentFaces) {
			if(trianglesToRemove.find(t) != trianglesToRemove.end()) { 
				alreadyDivided = true;
			}
		}
		if(!alreadyDivided) { // this edge can be used
			// get new point to add to mesh
			double x0 = points[edge[0]].getX();
			double x1 = points[edge[1]].getX();
			double y0 = points[edge[0]].getY();
			double y1 = points[edge[1]].getY();
			bool borderX = points[edge[0]].isBorderX() && x0 == x1;
			bool borderY = points[edge[0]].isBorderY() && y0 == y1;
			Point midpoint((x0 + x1) / 2, (y0 + y1) / 2, borderX, borderY);
			newPoints.push_back(midpoint); // note overall index of this point is numPoints + numDivided
			// handle triangles
			for(int t : incidentFaces) {
				trianglesToRemove.insert(t);
				// get opposite vertex
				int oppositeInd;
				for(int v = 0; v < 3; v++) {
					if(faces.at(t).at(v) != edge[0] && faces.at(t).at(v) != edge[1]) {
						oppositeInd = faces.at(t).at(v);
					}
				}
				// check for ccw orientation
				array<int, 2> orderedEdge = {edge[0], edge[1]};
				if(Triangle::getSignedArea(points + oppositeInd, points + edge[0], points + edge[1]) < 0) {
					orderedEdge = {edge[1], edge[0]};
				}
				newTriangles.push_back({oppositeInd, orderedEdge[0], numPoints + numDivided});
				newTriangles.push_back({oppositeInd, numPoints + numDivided, orderedEdge[1]});
			}
			numDivided++;
		}
		curIndex++;
	}

	cout << "subdivisions extracted" << endl;

	stepSize = originalStep;
	updateMesh(&newPoints, &newTriangles, &trianglesToRemove);
	updateApprox();
}

void Approx::updateMesh(vector<Point> *newPoints, vector<array<int, 3>> *newFaces, set<int> *discardedFaces) {
	vector<Point> oldPoints = getVertices();
	// free old memory
	cudaFree(points);
	cudaFree(triArr);
	// reallocate space
	int oldNumPoints = numPoints;
	numPoints += newPoints->size();
	int oldNumTri = numTri;
	numTri += newFaces->size() - discardedFaces->size();
	cudaMallocManaged(&points, numPoints * sizeof(Point));
	cudaMallocManaged(&triArr, numTri * sizeof(Triangle));

	// load points
	for(int i = 0; i < oldNumPoints; i++) {
		points[i] = oldPoints.at(i);
	}
	for(int i = 0; i < newPoints->size(); i++) {
		points[oldNumPoints + i] = newPoints->at(i);
	}

	// handle triangles
	// first remove triangles that were split by going in reverse order, since sets are sorted (?)
	for(auto f = discardedFaces->rbegin(); f != discardedFaces->rend(); f++) {
		faces.erase(faces.begin() + *f);
	}
	// add new triangles
	for(auto f = newFaces->begin(); f != newFaces->end(); f++) {
		faces.push_back(*f);
	}

	// update triArr
	for(int i = 0; i < numTri; i++) {
		array<int, 3> t = faces.at(i);
		triArr[i] = Triangle(points + t[0], points + t[1], points + t[2]);
		// ensure faces is ccw
		if(Triangle::getSignedArea(points + t[0], points + t[1], points + t[2]) < 0) {
			faces.at(i)[1] = t[2];
			faces.at(i)[2] = t[1];
		}
	}

	// update edges for next subdivision
	// TODO: don't clear the whole array, remove only the necessary parts
	edgeBelonging.clear();
	for(int i = 0; i < numTri; i++) {
		for(int j = 0; j < 3; j++) {
			array<int, 2> edge = {faces.at(i)[j], faces.at(i)[((j+1)%3)]};
			// ensure edges are ordered
			if(edge[0] > edge[1]) {
				edge[0] = edge[1];
				edge[1] = faces.at(i)[j];
			}
			if(edgeBelonging.find(edge) == edgeBelonging.end()) { // does not exist
				edgeBelonging[edge] = vector<int>();
			}
			edgeBelonging[edge].push_back(i);
		}		
	}
	gradX.clear();
	gradY.clear();
	tinyTriangles.clear();

    // re-allocate space for specific instance
    reallocateSpace(oldNumTri);
}

double Approx::getStep() {
    return stepSize;
}

vector<Point> Approx::getVertices() {
	vector<Point> vertices;
	for(int i = 0; i < numPoints; i++) {
		vertices.push_back(points[i]);
	}
	return vertices;
}

vector<array<int, 3>> Approx::getFaces() {
	return faces;
}

set<int> Approx::getTinyTriangles() {
	/*
	if(tinyTriangles.size() > 0) {
		cout << "small triangles: " << endl;
		for(int t : tinyTriangles) {
			cout << triArr[t];
		}
	}
	*/
	return tinyTriangles;
}

vector<Point> Approx::boundingBox() {
	vector<Point> corners;
	// create border frame around image
	corners.push_back(Point(-0.5,-0.5));
	corners.push_back(Point(-1,-1));
	corners.push_back(Point(maxX-0.5, -0.5));
	corners.push_back(Point(maxX, -1));
	corners.push_back(Point(maxX-0.5, maxY-0.5));
	corners.push_back(Point(maxX, maxY));
	corners.push_back(Point(-0.5, maxY-0.5));
	corners.push_back(Point(-1, maxY));
	return corners;
}

vector<array<int, 3>> Approx::boundingFaces() {
	vector<array<int, 3>> boundEdges;
	for(int i = 0; i < 8; i++) {
		boundEdges.push_back({i, (i+1)%8, (i+2)%8});
	}
	return boundEdges;
}
