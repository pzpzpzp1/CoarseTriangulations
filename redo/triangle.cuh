#ifndef triangle_cuh
#define triangle_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#endif

#include <math.h>
#include "point.cuh"
#include "segment.cuh"

/**
 * represent a triangle whose vertices are movable
 */

class Triangle {
	private:
		Point* vertices[3];
	public:
		// construct from three point pointers and orient in ccw direction
		Triangle(Point *a, Point *b, Point *c);
		CUDA_HOSTDEV double getArea();
		// get signed area based on the order of vertices
		// with ccw direction positive
		CUDA_HOSTDEV double getSignedArea();
		// get the change in area when the pth vertex is moving at velocity (vx, vy)
		double dA(int &p, double vx, double vy);
		// get the gradient in the x direction for pth vertex
		double gradX(int &p);
		// get the gradient in the y direction for pth vertex
		double gradY(int &p);

		// determine if triangle contains point p, including boundary points
		CUDA_DEV bool contains(Point &p);

		// return index of vertex opposite the middle length side
		int midVertex();
		// return length of longest side
		double maxLength();

		// get vertices of triangle, store in a, b, c
		void copyVertices(Point *a, Point *b, Point *c);

		// static signed area function
		CUDA_HOSTDEV static double getSignedArea(Point *a, Point *b, Point *c);

		friend ostream& operator<<(ostream& os, const Triangle &t);
		
		friend class Approx;
		friend class ParallelIntegrator;
		friend class Pixel;
};

#endif
