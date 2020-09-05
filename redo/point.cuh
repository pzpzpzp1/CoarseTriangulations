#ifndef point_cuh
#define point_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#endif

#include <math.h>
#include <iostream>
#include <iomanip>

using namespace std;

/**
 * represent a point (x,y) on the plane
 * mutable; allows perturbing vertices
 */

class Point {
	private:
		double x, y;
		// determine if point is on edge of image and thus cannot move
		bool borderX, borderY;
	public:
		CUDA_HOSTDEV Point();
		CUDA_HOSTDEV Point(double x, double y, bool borderX = false, bool borderY = false);
		CUDA_HOSTDEV double getX() const;
		CUDA_HOSTDEV double getY() const;
		// return true if point was constructed on a vertical image edge
		bool isBorderX() const;
		bool isBorderY() const;
		CUDA_HOSTDEV double distance(Point &other);
		void move(double deltaX, double deltaY);
		CUDA_DEV bool operator==(const Point &other) const;
		CUDA_DEV bool operator!=(const Point &other) const;

		friend ostream& operator<<(ostream& os, const Point &p);
};

#endif
