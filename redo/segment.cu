#include "segment.cuh"

// helper function for determining if t in [a, b] where order of a, b is unknown

__device__ Segment::Segment() {}

Segment::Segment(Point* a, Point* b) : endpoint1(a), endpoint2(b) {}


double Segment::length() {

	double x1 = endpoint1->getX();
	double y1 = endpoint1->getY();
	double x2 = endpoint2->getX();
	double y2 = endpoint2->getY();
	return pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5);
}

__device__ void Segment::unitNormal(double* nx, double* ny) {
	double deltaX = endpoint2->getX() - endpoint1->getX();
	double deltaY = endpoint2->getY() - endpoint1->getY();
	double unitX = deltaX / length();
	double unitY = deltaY / length();
	*nx = unitY;
	*ny = -unitX;
}

void Segment::scaledNormal(double* nx, double* ny) {
	double deltaX = endpoint2->getX() - endpoint1->getX();
	double deltaY = endpoint2->getY() - endpoint1->getY();
	*nx = deltaY / 2;
	*ny = -deltaX / 2;
}


__device__ void parametrize(Segment& e, Segment& f, double* t1, double* t2, double* det) {
	// parametrize and represent as matrix equation to be solved: 
	// t1 * x0 + (1-t1) * x1 = t2 * x2 + (1-t2) * x3
	// (x0-x1) * t1 + (x3-x2) * t2 = x3 - x1

	// note this originally was passed in to create a matrix
	// but dynamic matrix memory allocation costs time;
	// instead, use in this array form
	double arr[4];
	// first column (matches t1)
	arr[0] = e.endpoint1->getX() - e.endpoint2->getX();
	arr[2] = e.endpoint1->getY() - e.endpoint2->getY();
	// second column (matches t2)
	arr[1] = f.endpoint2->getX() - f.endpoint1->getX();
	arr[3] = f.endpoint2->getY() - f.endpoint1->getY();

	double determinant = arr[0] * arr[3] - arr[1] * arr[2];
	// target vector
	double targX = f.endpoint2->getX() - e.endpoint2->getX();
	double targY = f.endpoint2->getY() - e.endpoint2->getY();
	// scaled solution is adjugate of arr multiplied by target
	// adjugate is arr[3], -arr[1], -arr[2], arr[0]
	*t1 = arr[3] * targX - arr[1] * targY;
	*t2 = -arr[2] * targX + arr[0] * targY;
	*det = determinant;
}

__device__ bool isBetween(const double& t, const double& a, const double& b) {
	return (a <= t && t <= b) || (b <= t && t <= a);
}

__device__ bool Segment::intersection(Segment& other, Point* pt) {
	double t1 = 0;
	double t2 = 0;
	double det = 0;
	parametrize(*this, other, &t1, &t2, &det);
	bool detect = (det != 0) && isBetween(t1, 0, det) && isBetween(t2, 0, det);
	if (detect && pt) {
		double x = (endpoint1->getX() * t1 + endpoint2->getX() * (det - t1)) / det;
		double y = (endpoint1->getY() * t1 + endpoint2->getY() * (det - t1)) / det;
		*pt = Point(x, y);
	}
	return detect;
}


