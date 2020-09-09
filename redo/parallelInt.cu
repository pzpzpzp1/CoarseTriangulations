

#include "parallelInt.cuh"

// linear basis element derivatives
__device__ static const double phiU[3] = {-1, 1, 0};
__device__ static const double phiV[3] = {-1, 0, 1};

ParallelIntegrator::ParallelIntegrator() {
	threads2D = dim3(threadsX, threadsY);
}

bool ParallelIntegrator::initialize(Pixel *pix, int xMax, int yMax, ApproxType a, long long space, bool exact) {
	// steal references for easy access later
	pixArr = pix;
	approx = a;
	maxX = xMax;
	maxY = yMax;
	computeExact = exact;
	initialized = true;
	// allocate working computation space
	cudaMallocManaged(&arr, approx * sizeof(double *));
	for(int i = 0; i < approx; i++) {
		cudaMallocManaged(&(arr[i]), space * sizeof(double));
	}
	// less space needed for helper because it is only used for summing arr
	long long helperSpace = ceil(space / 512.0);
	cudaMallocManaged(&helper, helperSpace * sizeof(double));
	// the above operations may cause errors because so much memory is required
	cudaError_t error = cudaGetLastError();
  	if(error != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		cout << "An approximation of this quality is not possible due to memory limitations." << endl;
		return false;
	}
	cudaMallocManaged(&curTri, 3 * sizeof(Point));
	if(cudaGetLastError() != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		return false;
	}
	return true;
}

ParallelIntegrator::~ParallelIntegrator() {
	if(initialized) {
		for(int i = 0; i < approx; i++) {
			cudaFree(arr[i]);
		}
		cudaFree(arr);
		cudaFree(helper);
		cudaFree(curTri);
	}
}

__device__ void warpReduce(volatile double *sdata, unsigned int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

// kernel for sumArray
// compute the sum of an array arr with given size, in parallel
// with 1D thread/blocks, storing the result per block in result
__global__ void sumBlock(double *arr, int size, double *result) {
	__shared__ double partial[1024]; // hold partial results
	int tid = threadIdx.x;
	int ind = blockIdx.x * 2 * blockDim.x + tid;
	// load into partial result array
	if(ind + blockDim.x < size) {
		partial[tid] = arr[ind] + arr[ind + blockDim.x];
	} else if(ind < size) {
		partial[tid] = arr[ind];
	} else {
		partial[tid] = 0;
	}
	__syncthreads();

	// completely unroll the reduction
	if(tid < 512) {
		partial[tid] += partial[tid + 512];
	}
	__syncthreads();
	if(tid < 256) {
		partial[tid] += partial[tid + 256];
	}
	__syncthreads();
	if(tid < 128) {
		partial[tid] += partial[tid + 128];
	}
	__syncthreads();
	if(tid < 64) {
		partial[tid] += partial[tid + 64];
	}
	__syncthreads();

	// only one active warp at this point
	if(tid < 32) {
		warpReduce(partial, tid);
	}

	// write output for block to result
	if(tid == 0) {
		result[blockIdx.x] = partial[0];
	}
}

double ParallelIntegrator::sumArray(int size, int i) {
	int curSize = size; // current length of array to sum
	int numBlocks = (size + 2 * threads1D - 1) / (2 * threads1D);
	bool ansArr = true; // whether results are currently held in arr
	while(curSize > 1) {
		if(ansArr) {
			sumBlock<<<numBlocks, threads1D>>>(arr[i], curSize, helper);
		} else {
			sumBlock<<<numBlocks, threads1D>>>(helper, curSize, arr[i]);
		}
		cudaDeviceSynchronize();
		curSize = numBlocks;
		numBlocks = (numBlocks + 2 * threads1D - 1) / (2 * threads1D);
		ansArr = !ansArr;
	}
	// at this point the array has been summed
	if(ansArr) { // arr should hold the results
		return arr[i][0];
	}
	return helper[0];
}

// kernel for constantEnergyEval
// compute the energy of a single pixel on triangle triArr[t]
// weight by saliency value of pixel if salient
template<bool salient>
__global__ void pixConstantEnergyInt(Pixel *pixArr, int maxX, int maxY, Triangle tri, double color, double *results) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int ind = x * maxY + y; // index in pixArr;
	if(x < maxX && y < maxY) {
		double area = pixArr[ind].intersectionArea(tri);
		if(salient) area *= pixArr[ind].getSaliency();
		double diff = color - pixArr[ind].getColor();
		results[ind] = diff * diff * area;
	}
}

double ParallelIntegrator::constantEnergyExact(Triangle *tri, double color, bool salient) {
	dim3 numBlocks((maxX + threadsX - 1) / threadsX, (maxY + threadsY - 1) / threadsY);
	if(salient) {
		pixConstantEnergyInt<true><<<numBlocks, threads2D>>>(pixArr, maxX, maxY, *tri, color, arr[0]);
	} else {
		pixConstantEnergyInt<false><<<numBlocks, threads2D>>>(pixArr, maxX, maxY, *tri, color, arr[0]);
	}
	double answer = sumArray(maxX * maxY);
	return answer;
}

// kernel for constant energy approx
// using Point a as vertex point, sample ~samples^2/2 points inside the triangle with a triangular area element of dA
// NOTE: samples does not count endpoints along edge bc as the parallelograms rooted there lie outside the triangle
// maxY is for converting 2D pixel index to 1D index
template<bool salient>
__global__ void approxConstantEnergySample(Pixel *pixArr, int maxX, int maxY, Point *a, Point *b, Point *c, double color, double *results, double dA, int samples) {
	int u = blockIdx.x * blockDim.x + threadIdx.x; // component towards b
	int v = blockIdx.y * blockDim.y + threadIdx.y; // component towards c
	int ind = (2 * samples - u + 1) * u / 2 + v; // 1D index in results
	// this is because there are s points in the first column, s-1 in the next, etc. up to s - u + 1
	if(u + v < samples) {
		// get coordinates of this point using appropriate weights
		double x = (a->getX() * (samples - u - v) + b->getX() * u + c->getX() * v) / samples;
		double y = (a->getY() * (samples - u - v) + b->getY() * u + c->getY() * v) / samples;
		// find containing pixel
		int pixX = pixelRound(x, maxX);
		int pixY = pixelRound(y, maxY);
		double diff = color - pixArr[pixX * maxY + pixY].getColor();
		// account for points near edge bc having triangle contributions rather than parallelograms,
		// written for fast access and minimal branching
		double areaContrib = (u + v == samples - 1) ? dA : 2 * dA;
		if(salient) areaContrib *= pixArr[pixX * maxY + pixY].getSaliency();
		results[ind] = diff * diff * areaContrib;
	}
}

double ParallelIntegrator::constantEnergyApprox(Triangle *tri, double color, double ds, bool salient) {
	int i = tri->midVertex(); // vertex opposite middle side
	// ensure minVertex is copied into location curTri
	tri->copyVertices(curTri+((3-i)%3), curTri+((4-i)%3), curTri+((5-i)%3));
	// compute number of samples needed, using median number per side
	int samples = ceil(curTri[1].distance(curTri[2])/ds);
	// unfortunately half of these threads will not be doing useful work; no good fix, sqrt is too slow for triangular indexing
	dim3 numBlocks((samples + threadsX - 1) / threadsX, (samples + threadsY - 1) / threadsY);
	double dA = tri->getArea() / (samples * samples);
	if(salient) {
		approxConstantEnergySample<true><<<numBlocks, threads2D>>>(pixArr, maxX, maxY, curTri, curTri+1, curTri+2, color, arr[0], dA, samples);
	} else {
		approxConstantEnergySample<false><<<numBlocks, threads2D>>>(pixArr, maxX, maxY, curTri, curTri+1, curTri+2, color, arr[0], dA, samples);
	}
	cudaDeviceSynchronize();
	double answer = sumArray(samples * (samples + 1) / 2);
	return answer;
}

double ParallelIntegrator::constantEnergyEval(Triangle *tri, double color, double ds, bool salient) {
	// switch integration method based on exactnes required
	if(computeExact) {
		return constantEnergyExact(tri, color, salient);
	}
	return constantEnergyApprox(tri, color, ds, salient);
}

// kernel for constant line integral exact evaluation
// compute line integral of v dot n f ds for a single pixel and single triangle a, b, c when point b is moving
__global__ void pixConstantLineInt(Pixel *pixArr, int maxX, int maxY, Point *a, Point *b, Point *c, bool isX, double *results) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int ind = x * maxY + y;
	if (x < maxX && y < maxY) {
		double answer = 0;
		for(int i = 0; i < 2; i++) { // v dot n is nonzero only on a -- b and b -- c
			// extract segment and maintain ccw order for outward normal
			Segment seg = (i == 0) ? Segment(a, b) : Segment(b, c);
			Point *segEnd = (i == 0) ? a : c; // determine endpoint of seg that is not b
			double midX, midY; // to hold midpoint of segment intersection with this pixel
			double length = pixArr[ind].intersectionLength(seg, &midX, &midY);
			if(length != 0) {
				Point midpoint(midX, midY);
				// compute velocity at this point by scaling
				double distanceToVertex = midpoint.distance(*segEnd);
				double scale = distanceToVertex / seg.length(); // 1 if at b, 0 at opposite edge
				// velocity components
				double velX = (isX) ? scale : 0;
				double velY = scale - velX;
				// get unit normal values for this segment
				double nx, ny;
				seg.unitNormal(&nx, &ny);
				double vn = velX * nx + velY * ny; // average value of v * n
				answer += vn * length * pixArr[ind].getColor();
			}
		}
		results[ind] = answer;
	}
}

double ParallelIntegrator::lineIntExact(Triangle *tri, int pt, bool isX) {
	dim3 numBlocks((maxX + threadsX - 1) / threadsX, (maxY + threadsY - 1) / threadsY);
	tri->copyVertices(curTri, curTri+1, curTri+2);
	// compute integral in parallel based on function to integrate
	switch (approx) {
		case constant: {
			pixConstantLineInt<<<numBlocks, threads2D>>>(pixArr, maxX, maxY, curTri+((pt+2)%3), curTri+pt, curTri+((pt+1)%3), isX, arr[0]);
			break;
		}
		case linear: 
			cout << "Exact integration on linear approximations is not supported." << endl;
			exit(EXIT_FAILURE);
			break;
		case quadratic: // TODO
			break;
	}
	double answer = sumArray(maxX * maxY);
	return answer;
}

// kernel for constant line integral approximation
// compute line integral of v dot n f ds where point a is moving; 
// reverse determines if integral should be computed from a to b (false) or opposite
__global__ void constLineIntSample(Pixel *pixArr, int maxX, int maxY, Point *a, Point *b, bool reverse, bool isX, double *results, double ds, int samples) {
	int k = blockIdx.x * blockDim.x + threadIdx.x; // index along a to b
	if(k < samples) {
		// extract current point and containing pixel
		double x = (a->getX() * (samples - k) + b->getX() * k) / samples;
		double y = (a->getY() * (samples - k) + b->getY() * k) / samples;
		int pixX = pixelRound(x, maxX);
		int pixY = pixelRound(y, maxY);
		// velocity components
		double scale = ((double) samples - k) / samples; // 1 when k = 0 (evaluate at a) and 0 at b
		double velX = (isX) ? scale : 0;
		double velY = scale - velX;
		// extract unit normal, manually for the sake of speed
		double length = a->distance(*b); // length of whole segment
		// assume going from a to b first, want normal pointing right
		double nx = (b->getY() - a->getY()) / length;
		double ny = (a->getX() - b->getX()) / length;
		double vn = velX * nx + velY * ny; // value of v * n at this point
		// flip vn if normal is actually pointing the other way (integrate from b to a)
		if(reverse) vn *= -1;
		results[k] = vn * ds * pixArr[pixX * maxY + pixY].getColor();
	}
}

// kernel for constant line integral approximation, by taking average of all sample points
// phiA indicates whether the basis element at a is being integrated;
// if false, integrate element at b (element at c is zero on this segment)
__global__ void linearLineIntSample(Pixel *pixArr, int maxX, int maxY, Point *a, Point *b, bool reverse, bool isX, double *results, int samples, bool phiA) {
	int k = blockIdx.x * blockDim.x + threadIdx.x; // index along a to b
	if(k <= samples) {
		// extract current point and containing pixel
		double x = (a->getX() * (samples - k) + b->getX() * k) / samples;
		double y = (a->getY() * (samples - k) + b->getY() * k) / samples;
		int pixX = pixelRound(x, maxX);
		int pixY = pixelRound(y, maxY);
		// velocity components
		double scale = ((double) samples - k) / samples; // 1 when k = 0 (evaluate at a) and 0 at b
		double velX = (isX) ? scale : 0;
		double velY = scale - velX;
		// extract unit normal, manually for the sake of speed
		double length = a->distance(*b); // length of whole segment
		// assume going from a to b first, want normal pointing right
		double nx = (b->getY() - a->getY()) / length;
		double ny = (a->getX() - b->getX()) / length;
		double vn = velX * nx + velY * ny; // value of v * n at this point
		// flip vn if normal is actually pointing the other way (integrate from b to a)
		if(reverse) vn *= -1;
		// get value of phi
		double phi = (phiA) ? scale : 1 - scale; // since phi_j is linear, corresponds to scale
		results[k] = pixArr[pixX * maxY + pixY].getColor() * phi * vn;
	}
}

double ParallelIntegrator::lineIntApprox(Triangle *tri, int pt, bool isX, double ds, int basisInd) {
	// ensure pt is copied into the first slot of curTri
	tri->copyVertices(curTri+((3-pt)%3), curTri+((4-pt)%3), curTri+((5-pt)%3));
	// get number of samples for side pt, pt+1 and side pt, pt+2
	int samples[2];
	int numBlocks[2];
	for(int i = 0; i < 2; i++) {
		samples[i] = ceil(curTri->distance(curTri[i+1])/ds);
		numBlocks[i] = ceil(1.0 * samples[i] / threads1D);
	};
	double answer = 0; // integrate over both moving sides
	switch(approx) {
		case constant: {
			for(int i = 0; i < 2; i++) {
				double totalLength = curTri->distance(curTri[i+1]);
				// actual dx being used
				double dx = totalLength / samples[i];
				constLineIntSample<<<numBlocks[i], threads1D>>>(pixArr, maxX, maxY, curTri, curTri+i+1, (i==1), isX, arr[0], dx, samples[i]);
				cudaError_t cudaStatus = cudaDeviceSynchronize();
				answer += sumArray(samples[i]);
				cudaStatus = cudaDeviceSynchronize();
			}
			break;
		}
		case linear:
			// v phi_j is nonzero only if the line contains both vertex pt and basisInd
			if(basisInd == pt) {
				for(int i = 0; i < 2; i++) {
					double totalLength = curTri->distance(curTri[i+1]);
					// in case num samples is too small; ensure at least 2 points are sampled
					// (also prevent zero division error)
					samples[i] = max(samples[i], 2);
					linearLineIntSample<<<numBlocks[i], threads1D>>>(pixArr, maxX, maxY, curTri, curTri+i+1, (i==1), isX, arr[0], samples[i]-1, true);
					answer += totalLength * sumArray(samples[i]) / samples[i];
				}
			} else { // integrate along segment pt, basisInd
				int offset = (basisInd - pt + 3) % 3; // index of basisInd relative to pt
				int i = (offset + 1)%2; // index for this side's data in samples and numBlocks
				double totalLength = curTri->distance(curTri[offset]);
				// ensure at least 2 points are sampled
				samples[i] = max(samples[i], 2);
				linearLineIntSample<<<numBlocks[i], threads1D>>>(pixArr, maxX, maxY, curTri, curTri+offset, (offset==2), isX, arr[0], samples[i]-1, false);
				answer += totalLength * sumArray(samples[i]) / samples[i];
			}
			break;
		case quadratic:
			break;
	}
	return answer;
}

double ParallelIntegrator::lineIntEval(Triangle *tri, int pt, bool isX, double ds, int basisInd) {
	if(computeExact) {
		return lineIntExact(tri, pt, isX);
	}
	return lineIntApprox(tri, pt, isX, ds, basisInd);
}

// kernel for exact double integral
// compute double integral of f dA for a single pixel and single triangle triArr[t]
// pixArr is a 1D representation of image, where pixel (x, y) is at x * maxY + y
// reults holds the result for each pixel
__global__ void pixConstantDoubleInt(Pixel *pixArr, int maxX, int maxY, Triangle tri, double *results, ColorChannel channel) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int ind = x * maxY + y; // index in pixArr
	if(x < maxX && y < maxY) { // check bounds
		double area = pixArr[ind].intersectionArea(tri);
		results[ind] = area * pixArr[ind].getColor(channel);
	}
}

double ParallelIntegrator::doubleIntExact(Triangle *tri, ColorChannel channel) {
	dim3 numBlocks((maxX + threadsX -1) / threadsX, (maxY + threadsY -1) / threadsY);
	// compute integral in parallel based on function to integrate
	switch (approx) {
		case constant: {
			pixConstantDoubleInt<<<numBlocks, threads2D>>>(pixArr, maxX, maxY, *tri, arr[0], channel);
			break;
		}
		case linear: // TODO: fill out
			cout << "Exact integrals on non-constant approximations are not supported. Please change your approximation type." << endl;
			exit(EXIT_FAILURE);
			break;
		case quadratic: // TODO: fill out
			break;
	}
	double answer = sumArray(maxX * maxY);
	return answer;
}

// kernel for double integral approximation
// using Point a as vertex point, sample ~samples^2/2 points inside triangle with area element of dA
// for details see approxConstantEnergySample above
__global__ void constDoubleIntSample(Pixel *pixArr, int maxX, int maxY, Point *a, Point *b, Point *c, double *results, double dA, int samples, ColorChannel channel) {
	int u = blockIdx.x * blockDim.x + threadIdx.x; // component towards b
	int v = blockIdx.y * blockDim.y + threadIdx.y; // component towards c
	int ind = (2 * samples - u + 1) * u / 2 + v; // 1D index in results
	if(u + v < samples) {
		double x = (a->getX() * (samples - u - v) + b->getX() * u + c->getX() * v) / samples;
		double y = (a->getY() * (samples - u - v) + b->getY() * u + c->getY() * v) / samples;
		// find containing pixel
		int pixX = pixelRound(x, maxX);
		int pixY = pixelRound(y, maxY);
		double areaContrib = (u+v == samples - 1) ? dA : 2 * dA;
		results[ind] = pixArr[pixX * maxY + pixY].getColor(channel) * areaContrib;
	}
}

// compute integral f phi_j dA by barycentric sampling
// using pts[0] as vertex point; store values in results[j]
__global__ void linearDoubleIntSample(Pixel *pixArr, int maxX, int maxY, Point *pts, double **results, double dA, int samples, ColorChannel channel) {
	int u = blockIdx.x * blockDim.x + threadIdx.x; // component towards pts[1]
	int v = blockIdx.y * blockDim.y + threadIdx.y; // component towards pts[2]
	int ind = (2 * samples - u + 1) * u / 2 + v; // 1D index in results[j]
	if(u + v < samples) {
		// extract coordinates at this sample point
		double x = (pts[0].getX() * (samples - u - v) + pts[1].getX() * u + pts[2].getX() * v) / samples;
		double y = (pts[0].getY() * (samples - u - v) + pts[1].getY() * u + pts[2].getY() * v) / samples;
		// get color of containing pixel
		double color = pixArr[pixelRound(x, maxX) * maxY + pixelRound(y, maxY)].getColor(channel);
		// scale fdA by 1/samples to avoid multiple divisions in FEM basis computation later
		double fdA = dA * color / samples;
		// area element is a parallelogram except for triangular contributions at the opposite edge
		// when u + v == samples - 1
		if(u + v < samples - 1) fdA *= 2;
		// second factor is (scaled) FEM basis value at this point
		results[0][ind] = fdA * (samples - u - v);
		results[1][ind] = fdA * u;
		results[2][ind] = fdA * v;
	}
}

void ParallelIntegrator::doubleIntApprox(Triangle *tri, double ds, double *result, ColorChannel channel) {
	// extract number of samples
	int i = tri->midVertex();
	// copy middle vertex into curTri[0]
	tri->copyVertices(curTri+((3-i)%3), curTri+((4-i)%3), curTri+((5-i)%3));
	int samples = ceil(curTri[1].distance(curTri[2])/ds);
	dim3 numBlocks((samples + threadsX - 1) / threadsX, (samples + threadsY - 1) / threadsY);
	double dA = tri->getArea() / (samples * samples);
	switch(approx) {
		case constant: {
			constDoubleIntSample<<<numBlocks, threads2D>>>(pixArr, maxX, maxY, curTri, curTri+1, curTri+2, arr[0], dA, samples, channel);
			break;
		}
		case linear: {
			linearDoubleIntSample<<<numBlocks, threads2D>>>(pixArr, maxX, maxY, curTri, arr, dA, samples, channel);
			break;
		}
	}
	cudaDeviceSynchronize();

	// store results into result in order aligning with tri
	// (result[j] is integral of phi_j, which is 1 on tri.vertices[j] and 0 on the other vertices)
	for(int j = 0; j < approx; j++) {
		int relativeBasis = (j + approx - i) % approx; // with reference to vertices of curTri; when j == i, this is 0
		result[j] = sumArray(samples * (samples + 1) / 2, relativeBasis);
	}
}

void ParallelIntegrator::doubleIntEval(Triangle *tri, double ds, double *result, ColorChannel channel) {
	if(computeExact) {
		*result = doubleIntExact(tri, channel);
	} else {
		doubleIntApprox(tri, ds, result, channel);
	}
}

// kernel function for linearEnergyApprox
// assuming point a as vertex and matching k0, k1, k2 to a, b, c,
// sample (f - sum k_i phi_i)^2 over the triangle
// weighted by saliency if salient
template<bool salient>
__global__ void approxLinearEnergySample(Pixel *pixArr, int maxX, int maxY, Point *a, Point *b, Point *c, double k0, double k1, double k2, double *results, double dA, int samples) {
	int u = blockIdx.x * blockDim.x + threadIdx.x; // component towards b
	int v = blockIdx.y * blockDim.y + threadIdx.y; // component towards c
	int ind = (2 * samples - u + 1) * u / 2 + v; // 1D index in results
	// this is because there are s points in the first column, s-1 in the next, etc. up to s - u + 1
	if(u + v < samples) {
		// get coordinates of this point using appropriate weights
		double x = (a->getX() * (samples - u - v) + b->getX() * u + c->getX() * v) / samples;
		double y = (a->getY() * (samples - u - v) + b->getY() * u + c->getY() * v) / samples;
		// find containing pixel
		int pixX = pixelRound(x, maxX);
		int pixY = pixelRound(y, maxY);
		// find color at this point using standard transform
		double diff = (k0 * (samples - u - v) + k1 * u + k2 * v) / samples - pixArr[pixX * maxY + pixY].getColor();
		// account for points near edge bc having triangle contributions rather than parallelograms,
		// written for fast access and minimal branching
		double areaContrib = (u + v == samples - 1) ? dA : 2 * dA;
		if(salient) {
			areaContrib *= pixArr[pixX * maxY + pixY].getSaliency();
		}
		results[ind] = diff * diff * areaContrib;
	}
}

double ParallelIntegrator::linearEnergyApprox(Triangle *tri, double *coeffs, double ds, bool salient) {
	int i = tri->midVertex(); // vertex opposite middle side
	// curTri[0] = tri.vertices[i]
	tri->copyVertices(curTri + ((3-i)%3), curTri + ((4-i)%3), curTri + ((5-i)%3));
	// compute number of samples needed, using median number per side
	int samples = ceil(curTri[1].distance(curTri[2])/ds);
	// unfortunately half of these threads will not be doing useful work; no good fix, sqrt is too slow for triangular indexing
	dim3 numBlocks((samples + threadsX - 1) / threadsX, (samples + threadsY - 1) / threadsY);
	double dA = tri->getArea() / (samples * samples);
	if(salient) {
		approxLinearEnergySample<true><<<numBlocks, threads2D>>>(pixArr, maxX, maxY, curTri, curTri + 1, curTri + 2,
			coeffs[i], coeffs[(i+1)%3], coeffs[(i+2)%3], arr[0], dA, samples);
	} else {
		approxLinearEnergySample<false><<<numBlocks, threads2D>>>(pixArr, maxX, maxY, curTri, curTri + 1, curTri + 2,
			coeffs[i], coeffs[(i+1)%3], coeffs[(i+2)%3], arr[0], dA, samples);
	}
	double answer = sumArray(samples * (samples + 1) / 2);
	return answer;
}

// kernel function for computing integral of 2A_T f d(phi_j) dA when pts[0] is moving at (1,0)
// phiU, phiV indicate d phi/du, d phi/dv and dA_x is the area gradient
__global__ void linearImageGradientX(Pixel *pixArr, int maxX, int maxY, Point *pts, double **results, double dA, double dA_x, int samples) {
	int uInd = blockIdx.x * blockDim.x + threadIdx.x; // component towards pts[1]
	int vInd = blockIdx.y * blockDim.y + threadIdx.y; // component towards pts[2]
	double u = (double) uInd / samples;
	double v = (double) vInd / samples;
	int ind = (2 * samples - uInd + 1) * uInd / 2 + vInd; // 1D index in results
	if(uInd + vInd < samples) {
		// get coordinates of this point using appropriate weights
		double x = (pts[0].getX() * (samples - uInd - vInd) + pts[1].getX() * uInd + pts[2].getX() * vInd) / samples;
		double y = (pts[0].getY() * (samples - uInd - vInd) + pts[1].getY() * uInd + pts[2].getY() * vInd) / samples;
		// compute du/dt, dv/dt at this point (scaled by 2A_T)
		double du = y - pts[2].getY() - 2 * u * dA_x;
		double dv = pts[1].getY() - y - 2 * v * dA_x;
		// find color of containing pixel
		double color = pixArr[pixelRound(x, maxX) * maxY + pixelRound(y, maxY)].getColor();
		// account for points near opposite edge having triangle contributions rather than parallelograms
		double fdA = (uInd + vInd == samples - 1) ? color * dA : 2 * color * dA;
		// compute all three basis element contributions
		results[0][ind] = fdA * (phiU[0] * du + phiV[0] * dv);
		results[1][ind] = fdA * (phiU[1] * du + phiV[1] * dv);
		results[2][ind] = fdA * (phiU[2] * du + phiV[2] * dv);
	}
}

// same kernel function but when pts[0] is moving at (0,1)
__global__ void linearImageGradientY(Pixel *pixArr, int maxX, int maxY, Point *pts, double **results, double dA, double dA_y, int samples) {
	int uInd = blockIdx.x * blockDim.x + threadIdx.x; // component towards pts[1]
	int vInd = blockIdx.y * blockDim.y + threadIdx.y; // component towards pts[2]
	double u = (double) uInd / samples;
	double v = (double) vInd / samples;
	int ind = (2 * samples - uInd + 1) * uInd / 2 + vInd; // 1D index in results
	if(uInd + vInd < samples) {
		// get coordinates of this point using appropriate weights
		double x = (pts[0].getX() * (samples - uInd - vInd) + pts[1].getX() * uInd + pts[2].getX() * vInd) / samples;
		double y = (pts[0].getY() * (samples - uInd - vInd) + pts[1].getY() * uInd + pts[2].getY() * vInd) / samples;
		// find du/dt, dv/dt at this point (scaled by 2A_T)
		double du = pts[2].getX() - x - 2 * u * dA_y;
		double dv = x - pts[1].getX() - 2 * v * dA_y;
		// find color of containing pixel
		double color = pixArr[pixelRound(x, maxX) * maxY + pixelRound(y, maxY)].getColor();
		// account for points near opposite edge having triangle contributions rather than parallelograms
		double fdA = (uInd + vInd == samples - 1) ? color * dA : 2 * color * dA;
		// compute all three basis element contributions
		results[0][ind] = fdA * (phiU[0] * du + phiV[0] * dv);
		results[1][ind] = fdA * (phiU[1] * du + phiV[1] * dv);
		results[2][ind] = fdA * (phiU[2] * du + phiV[2] * dv);
	}
}

void ParallelIntegrator::linearImageGradient(Triangle *tri, int pt, bool isX, double ds, double *result) {
	// copy pt into curTri[0]
	tri->copyVertices(curTri+((3-pt)%3), curTri+((4-pt)%3), curTri+((5-pt)%3));
	// extract number of samples
	int i = tri->midVertex();
	int samples = ceil(((tri->vertices[(i+1)%3])->distance(*(tri->vertices[(i+2)%3])))/ds);
	dim3 numBlocks((samples + threadsX - 1) / threadsX, (samples + threadsY - 1) / threadsY);
	double dA = tri->getArea() / (samples * samples);
	double dA_x = tri->gradX(pt);
	double dA_y = tri->gradY(pt);
	if(isX) {
		linearImageGradientX<<<numBlocks, threads2D>>>(pixArr, maxX, maxY, curTri, arr, dA, dA_x, samples);
	} else {
		linearImageGradientY<<<numBlocks, threads2D>>>(pixArr, maxX, maxY, curTri, arr, dA, dA_y, samples);
	}
	for(int j = 0; j < approx; j++) {
		int relativeBasis = (j - pt + approx) % approx; // align curTri with ordering of basis elements
		result[j] = sumArray(samples * (samples + 1) / 2, relativeBasis) / (2 * tri->getArea());
	}
}