#include "triangle.cuh"

// custom rounding function to support needed pixel rounding

Triangle::Triangle(Point *a, Point *b, Point *c) {
	vertices[0] = a;
	vertices[1] = b;
	vertices[2] = c;
	if(getSignedArea() < 0) { // reverse direction
		vertices[1] = c;
		vertices[2] = b;
	}
}

double Triangle::getSignedArea() {
	double ax = vertices[0]->getX();
	double ay = vertices[0]->getY();
	double bx = vertices[1]->getX();
	double by = vertices[1]->getY();
	double cx = vertices[2]->getX();
	double cy = vertices[2]->getY();
	// determinant of matrix [bx - ax, cx - ax, by - ay, cy - ay] / 2
	return ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2;
}

double Triangle::getArea() {
	double signedArea = getSignedArea();
	if (signedArea < 0) {
		return -signedArea;
	}
	return signedArea;
}

double Triangle::dA(int &p, double vx, double vy) {
	// first extract the other two endpoints; note order matters
	Point* edgePoints[2];
	// retrieve in ccw order
	edgePoints[0] = vertices[(p+1)%3];
	edgePoints[1] = vertices[(p+2)%3];
	// change is -velocity dot edge normal of length |e|/2
	Segment opposite(edgePoints[0], edgePoints[1]);
	// get normal to segment
	double nx, ny;
	opposite.scaledNormal(&nx, &ny);
	// return negative of dot product
	return -(vx * nx + vy * ny);
}

double Triangle::gradX(int &p) {
	return dA(p, 1, 0);
}

double Triangle::gradY(int &p) {
	return dA(p, 0, 1);
}

__device__ bool Triangle::contains(Point &p) {
	// p is inside the triangle iff the orientations of the triangles
	// with vertices (vertices[i], vertices[i+1], p) are all ccw
	for(int i = 0; i < 3; i++) {
		if (Triangle::getSignedArea(vertices[i], vertices[(i+1)%3], &p) < 0) {
			return false;
		}
	}
	return true;
}

int Triangle::midVertex() {
	double distances[3];
	for(int i = 0; i < 3; i++) {
		// get length of opposite side
		distances[i] = vertices[(i+1)%3]->distance(*vertices[(i+2)%3]);
	}
	for(int i = 0; i < 3; i++) {
		if(distances[i] >= min(distances[(i+1)%3], distances[(i+2)%3]) && 
			distances[i] <= max(distances[(i+1)%3], distances[(i+2)%3])) return i;
	}
	throw runtime_error("should not get here");
	return -1; // to make compiler happy
}

double Triangle::maxLength() {
	double distance = 0;
	for(int i = 0; i < 3; i++) {
		distance = max(distance, vertices[(i+1)%3]->distance(*vertices[(i+2)%3]));
	}
	return distance;
}

void Triangle::copyVertices(Point *ptrA, Point *ptrB, Point *ptrC) {
	*ptrA = *vertices[0];
	*ptrB = *vertices[1];
	*ptrC = *vertices[2];
}

double Triangle::getSignedArea(Point *a, Point *b, Point *c) {
	double ax = a->getX();
	double ay = a->getY();
	double bx = b->getX();
	double by = b->getY();
	double cx = c->getX();
	double cy = c->getY();
	return ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2;
}

ostream& operator<<(ostream& os, const Triangle &t) {
	os << "Triangle ";
	for(Point *ptr : t.vertices) {
		os << *ptr << " ";
	}
	os << "\n";
	return os;
}
