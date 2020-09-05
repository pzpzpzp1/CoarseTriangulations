#include "pixel.cuh"

// constants for converting rgb to grayscale
const double RED_LUMINANCE = 0.2126;
const double GREEN_LUMINANCE = 0.7152;
const double BLUE_LUMINANCE = 0.0722;

// get luminance of an rgb value by standard transformation
int getLuminance(int r, int g, int b) {
	return round(r * RED_LUMINANCE + g * GREEN_LUMINANCE + b * BLUE_LUMINANCE);
}

// helper functions

// determine whether a value has fractional part 1/2
// (used to determine whether point is a pixel corner)
__device__ bool isHalfInteger(double x) {
	return (x - floor(x) == 0.5);
}

// determine whether two points are "essentially" equal (floating point error)
__device__ bool approxEqual(Point &a, Point &b, double tolerance = 1e-12) {
	return (a.distance(b) < tolerance);
}

// compute (unsigned) area of the polygon enclosed by points,
// where edegs of the polygon are given by points[i] -- points[i+1]
__device__ double shoelace(Point *points, int &size) {
	if (size < 3) {
		return 0;
	}
	double area = 0;
	for(int i = 0; i < size; i++) {
		double x0 = points[i].getX();
		double y0 = points[i].getY();
		double x1 = points[(i+1)%size].getX();
		double y1 = points[(i+1)%size].getY();
		area += (x0 * y1 - x1 * y0);
	}
	// in practice points is ccw
	// up to floating point errors that don't affect area
	//assert(area >= 0);
	return area/2;
}

// compute integral of x over polygon points and store it in totalX, sim for y
// center is a reference point inside the pixel; even if it lies outside the polygon,
// using signed areas means the result will still be correct
__device__ void integrateXY(double *totalX, double *totalY, Point *points, int &size, Point &center) {
	double sumX = 0;
	double sumY = 0;
	for(int i = 0; i < size; i++) {
		// average value over a triangle is just the centroid
		double centroidX = (points[i].getX() + points[(i+1)%size].getX() + center.getX())/3;
		double centroidY = (points[i].getY() + points[(i+1)%size].getY() + center.getY())/3;
		double triangleArea = Triangle::getSignedArea(&center, &points[i], &points[(i+1)%size]);
		// weight the average
		sumX += centroidX * triangleArea;
		sumY += centroidY * triangleArea;
	}
	*totalX = sumX;
	*totalY = sumY;
}

// compute average values of x, y over the polygon enclosed by points
// and put them in the given variables
// center is again a reference point
__device__ void averageXY(double *avgX, double *avgY, Point *points, int &size, Point &center) {
	double totalX;
	double totalY;
	integrateXY(&totalX, &totalY, points, size, center);
	double totalArea = shoelace(points, size);
	*avgX = totalX / totalArea;
	*avgY = totalY / totalArea;
}

Pixel::Pixel(int x_, int y_, int c) : x(x_), y(y_) {
	corners[0] = Point(x-0.5, y-0.5);
	corners[1] = Point(x+0.5, y-0.5);
	corners[2] = Point(x+0.5, y+0.5);
	corners[3] = Point(x-0.5, y+0.5);
	for(int i = 0; i < 4; i++) {
		colors[i] = c;
	}
}

Pixel::Pixel(int x_, int y_, int r, int g, int b) : x(x_), y(y_) {
	corners[0] = Point(x-0.5, y-0.5);
	corners[1] = Point(x+0.5, y-0.5);
	corners[2] = Point(x+0.5, y+0.5);
	corners[3] = Point(x-0.5, y+0.5);
	colors[0] = r;
	colors[1] = g;
	colors[2] = b;
	colors[3] = getLuminance(r, g, b);
}

double Pixel::getColor(ColorChannel channel) {
	return colors[channel];
}

__device__ double Pixel::getSaliency() {
	return saliency;
}

void Pixel::setSaliency(double s) {
	assert(s >= 0);
	saliency = s;
}

__device__ bool Pixel::containsPoint(Point &p) {
	double px = p.getX();
	double py = p.getY();
	return (-0.5+x <= px && px <= 0.5+x) && (-0.5+y <= py && py <= 0.5+y);
}

__device__ double Pixel::intersectionLength(Segment &e, double *xVal, double *yVal) {
	Point intersections[2]; // hold intersections
	int numPts = 0; // track number of intersection points detected thus far
	Point intersectionPoint; // hold current potential intersection point
	for(int i = 0; i < 4; i++) {
		// retrieve a side of the pixel; at most two will have an 
		// intersection unless intersection is at corners
		Segment side(&corners[i], &corners[(i+1)%4]);
		bool collision = side.intersection(e, &intersectionPoint);
		if (collision) {
			bool isNewPoint = true; // whether this intersection is a new distinct point
			for(int i = 0; i < numPts; i++) {
				if(approxEqual(intersections[i], intersectionPoint)) {
					isNewPoint = false;
				}
			}
			if (isNewPoint) {
				intersections[numPts] = intersectionPoint;
				numPts++;
			}
		}
	}
	// handle segment endpoints potentially inside the pixel
	if (numPts < 2) {
		Point start = *(e.endpoint1);
		Point end = *(e.endpoint2);
		if (containsPoint(start)) {
			intersections[numPts] = start;
			numPts++;
		}
		if (containsPoint(end)) {
			intersections[numPts] = end;
			numPts++;
		}
	}
	if (numPts < 2) {
		return 0;
	}
	Segment contained(&intersections[0], &intersections[1]);
	// check for null pointers, assign midpoint coords
	if (xVal && yVal) {
		*xVal = (intersections[0].getX() + intersections[1].getX())/2;
		*yVal = (intersections[0].getY() + intersections[1].getY())/2;
	}
	return contained.length();
}

__device__ double Pixel::intersectionArea(Triangle t, Point* polygon, int *size) {
	Point center(x, y); // center of this pixel
	int numPoints = 0; // track number of points in polygon
	Point boundary[8]; // there should only be max 8 points on the boundary,
	int inInd; // index of some triangle vertex that lies inside pixel (may not exist)
	Segment triangleSides[3]; // hold sides of triangle

	// goal: compute boundary of the intersection

	for(int i = 0; i < 3; i++) {
		triangleSides[i] = Segment(t.vertices[i], t.vertices[(i+1)%3]);
		// add triangle vertices which may be inside the pixel, but don't add corners
		bool isCorner = isHalfInteger(t.vertices[i]->getX()) && isHalfInteger(t.vertices[i]->getY());
        if (!isCorner && containsPoint(*(t.vertices[i]))) {
            inInd = i;
			boundary[numPoints] = *(t.vertices[i]);
			numPoints++;
		}
	}

    // determine corner to start so as to preserve ccw property
    int start = 0;
    // do this by starting from a corner outside the triangle (if it exists);
	// if it doesn't exist start will stay at 0
    for(int i = 0; i < 4; i++) {
        // additionally, if there is exactly one point inside the triangle, make sure to start
        // at a corner on the same side of the interior point so that the first edge
        // interior point -- intersection point is correct (avoid issues of pixel corners inside
        // the triangle being non-adjacent)
        bool safelyOriented = (numPoints != 1) || 
			(Triangle::getSignedArea(corners + i, t.vertices[(inInd+1)%3], t.vertices[(inInd+2)%3]) >= 0);
        if (safelyOriented && !t.contains(corners[i])) {
			start = i;
			break; // including this line gives a 25% speed increase
		}
	}
    for(int i = 0; i < 4; i++) {
        // first determine if corner of pixel is inside
        Point corner = corners[(i+start) % 4];
		Segment side(corners + ((i+start)%4), corners + ((i+start+1)%4));
		// OPTIMIZATION: BRANCHING HERE; unavoidable?
        if (t.contains(corner)) {
			boundary[numPoints] = corner;
			numPoints++;
		}
        // determine intersections with side (i, i+1)
		Point sideIntersections[2];
		int intersectNum = 0; // track index in sideIntersections
		Point intersectionPoint; // track current intersection point
        for(Segment e : triangleSides) {
			// true if intersection exists
			bool collision = side.intersection(e, &intersectionPoint);
			if (collision) {
                // check to see if this point is already accounted for by corners
                // or by triangle vertices; if it isn't exactly equal it won't contribute to area
                // (and the lack of exact equality is likely due to floating point error)
                if (!approxEqual(intersectionPoint, corner) && !approxEqual(intersectionPoint, corners[(i+start+1)%4])) {
                    bool isVertex = false;
                    for(Point *tVertex : t.vertices) {
                        if (approxEqual(intersectionPoint, *tVertex)) {
                            isVertex = true;
                        }
                    }
                    if (!isVertex) {
						sideIntersections[intersectNum] = intersectionPoint;
						intersectNum++;
                    }
                }
            }
		}
		/*
		if(intersectNum > 2) {
			printf("INCORRECT INTERSECTION NUM: %d\n", intersectNum);
			for(int k = 0; k < intersectNum; k++) {
				printf("(%f, %f)\n", sideIntersections[k].getX(), sideIntersections[k].getY());
			}
			printf("END INCORRECT\n");
		}
		*/
        // note a triangle can intersect a given side at most twice
        assert(intersectNum <= 2);
		// handle normal case where there is only one intersection with this side
        if (intersectNum == 1) {
			boundary[numPoints] = sideIntersections[0];
			numPoints++;
        } else if (intersectNum == 2) {
            double signedArea = Triangle::getSignedArea(&center, &sideIntersections[0], &sideIntersections[1]);
            // if signedArea == 0, sideIntersections must contain two of the same point
            // which means one vertex of the triangle is on the side; this has
			// already been accounted for and shouldn't happen because of vertex check
			if(signedArea != 0) {
				numPoints += 2;
				int nearestInd = (signedArea < 0) ? 1 : 0; // first point of sideIntersections in ccw order
				boundary[numPoints-2] = sideIntersections[nearestInd];
				boundary[numPoints-1] = sideIntersections[1 - nearestInd];
			}
		}
    }
    // check for null pointer
    if (polygon && size) {
        polygon = boundary;
		*size = numPoints;
    }
    return shoelace(boundary, numPoints);
}

__device__ double Pixel::approxArea(Triangle &t, int n) {
	// width of a square in the lattice grid;
	// this ensures n points per side
	double ds = 1.0/(n-1);
	int numPoints = 0; // number of lattice points inside the triangle
	// weight boundary points by 1/2, as in Pick's
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			double xval = x - 0.5 + ds * i;
			double yval = y - 0.5 + ds * j;
			bool contains = true; // whether this point is contained
			bool strictly = true;
			// iterate over vertices
			for(int v = 0; v < 3; v++) {
				int w = (v+1)%3; // avoid doing slow computation twice
				// these accesses seem slow but there doesn't seem to be a better way
				double bx = t.vertices[v]->getX() - xval;
				double by = t.vertices[v]->getY() - yval;
				double cx = t.vertices[w]->getX() - xval;
				double cy = t.vertices[w]->getY() - yval;
				double sign = bx * cy - cx * by;
				// branch divergence here :( nothing seems to speed it up?
				if(sign < 0) {
					contains = false;
					strictly = false;
					break;
				}
				if(sign == 0) strictly = false;
			}
			// count boundary points once and interior points twice
			numPoints += contains + strictly;
		}
	}
	// approximate area
	return numPoints / (2.0 * n * n);
}


int pixelRound(double x, int bound) {
	int floor = (int) x;
	if (abs(x - floor) <= 0.5) {
		return floor;
	} else if (x > 0) {
		return min(floor + 1, bound - 1);
	}
	return 0;
}