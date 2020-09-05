#ifndef segment_cuh
#define segment_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#endif

#include <math.h>
#include "point.cuh"

using namespace std;
/**
 * represent a point (point1, point2) in the plane
 */

class Segment {
	private:
		Point *endpoint1, *endpoint2;
	public:
		CUDA_DEV Segment(); // this just exists to create arrays // original
		CUDA_HOSTDEV Segment(Point *a, Point *b);
		
		CUDA_HOSTDEV double length();

		// get the 2x1 unit normal to this segment which lies on the right
		// going from endpoint 1 to endpoint 2, setting values in nx, ny
		CUDA_DEV void unitNormal(double *nx, double *ny);
		// return the 2x1 normal to this segment that has length |segment|/2
		void scaledNormal(double *nx, double *ny);

		// helper function that determines parameters at intersection point of e and f,
		// storing as t1, t2; intersection is x0 * (t1 / det) + x1 * (1 - t1/det)
		friend CUDA_DEV void parametrize(Segment &e, Segment &f, double *t1, double *t2, double *det);

		// combine intersection determination and finding point of intersection;
		// return true if this segment intersects other in exactly one point,
		// store intersection point in pt if they intersect (undefined behavior if
		// intersection does not exist or if segments overlap with positive length)
		CUDA_DEV bool intersection(Segment& other, Point* pt = NULL);

		
		friend class Pixel;
		
};

#endif 
