#ifndef pixel_cuh
#define pixel_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#endif

#include <assert.h>
#include "point.cuh"
#include "segment.cuh"
#include "triangle.cuh"
using namespace std;

// enumeration for colors where gray represents luminance (grayscale value)
enum ColorChannel{RED, GREEN, BLUE, GRAY};

/**
 * class representing an image pixel as a square, with pixel
 * (x,y) being a unit square centered at (x, y)
 */

class Pixel {
	private:
		int colors[4]; // represent rgb and luminance of pixel from 0 to 255
		int x, y; // center of pixel
		Point corners[4]; // corners of pixel in ccw order
		double saliency = 1.0; // represent saliency value at the center of this pixel
	public:
		// create a grayscale pixel centered at (x, y) with luminance c
		Pixel(int x, int y, int c);
		// create a color pixel centered at (x, y) with given rgb values 
		Pixel(int x, int y, int r, int g, int b);
		// get color of a given channel, defaulting to grayscale
		CUDA_HOSTDEV double getColor(ColorChannel channel = GRAY);
		// get saliency
		CUDA_DEV double getSaliency();
		// set saliency value; s must be >= 0
		void setSaliency(double s);
		// compute the length of the intersection of Segment e
		// with this pixel (may be 0); if x, y are given, store the coordinates
		// of the midpoint of the segment in x, y; DOES NOT SET x, y if length is 0
		CUDA_DEV double intersectionLength(Segment &e, double *x = NULL, double *y = NULL);
		// return true if this pixel contains point p (inside or on the boundary)
		CUDA_DEV bool containsPoint(Point &p);

		// compute area of the intersection of this pixel with Triangle t (may be 0);
		// store the boundary of the intersection in polygon if available along
		// with the size of the boundary
		CUDA_DEV double intersectionArea(Triangle t, Point* polygon = NULL, int* size = NULL);
		CUDA_DEV double approxArea(Triangle &t, int n = 5);
};

// helper function for rounding to pixel values
// where -0.5 <= x <= bound - 0.5
// upper bounded by bound in case of floating point error
CUDA_HOSTDEV int pixelRound(double x, int bound);

#endif
