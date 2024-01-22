#include<stdio.h>
#include "goGpuDet.h"

//generate a random number

__device__ Int64_t Uint64(int* inp, int endd) {
	Int64_t b[8];
	for (int i = 0;i < 8;i++) {
		b[i] = (Int64_t)inp[endd-8 + i];
	}
	int end = 8;
	Int64_t x = (Int64_t)b[end - 1] | (Int64_t)b[end - 2] << 8 | (Int64_t)b[end - 3] << 16 | (Int64_t)b[end - 4] << 24 |
		(Int64_t)b[end - 5] << 32 | (Int64_t)b[end - 6] << 40 | (Int64_t)b[end - 7] << 48 | (Int64_t)b[end - 8] << 56;
	return x;
}

//generate the random matrix
__device__ void generateRandomMat_Det(int* hashBytes, double* retD,int hashsize,int msize, int height,int blockindex) {
 	double matrix[900];
 	randObj rObj;
	int dim = msize;
	Int64_t n = 0LL;
	Int64_t hashSeed[4] = { 0,0,0,0 };
	Int64_t curNum = 0LL;
	
	int tid = blockindex;
	

	int start = tid * 32;

	
	hashSeed[0] = Uint64(hashBytes, 8+start);
	hashSeed[1] = Uint64(hashBytes, 16+start);
	hashSeed[2] = Uint64(hashBytes, 24+start);
	hashSeed[3] = Uint64(hashBytes, 32+start);

	for (int i = 0; i < dim; ++i) {
		curNum ^= hashSeed[i % 4];

		
		if (height >= EmeryForkHeight) {

			NewSource_EmeryFork(curNum,&rObj);
		}
		else {
			NewSource(curNum,&rObj);
		}

		for (int j = 0; j < dim; ++j) {
			n = (1ULL << 63) - 1;

			curNum = rObj_Int63n(n, &rObj);

			matrix[i*dim+j] = (double)rObj_Int63n(3, &rObj);

		}
		


	}
	
	double tmpd = (double)glb_determinant(matrix,dim);
	
	retD[tid] = tmpd;
	
}

//gausian elimination method for computing determinants
__device__ double glb_determinant(double* matrix,int dim) {
	
	int n = dim;
    double a[matrixSize];
	

	
	for (int i = 0;i < dim*dim; ++i) {
		a[i] = matrix[i];

	}
	double det = (double)1.0;
	double temp;
	int i, k, j;
	for (i = 0; i < n; ++i) {
		k = i;
		for (j = i + 1; j < n; ++j) {
			if (fabs(a[j*dim+i]) > fabs(a[k*dim+i])) {
				k = j;
			}
		}
		if (fabs(a[k*dim+i]) < EPS) {

			det = 0;
			break;
		}
		if (i != k) {
			for (j = i;j < n;++j) {
				temp = a[i*dim+j];
				a[i*dim+j] = a[k*dim+j];
				a[k*dim+j] = temp;
			}


			det = -det;
		}

		for (j = i + 1; j < n; ++j)
			a[i*dim+j] = (double)(a[i*dim+j] / a[i*dim+i]);
		for (j = i + 1; j < n; ++j)

			for (int g = i + 1; g < n; ++g)
				a[j*dim+g] = (double)(a[j*dim+g] - a[i*dim+g] * a[j*dim+i]);


	}

	for (int i = 0;i < n;++i) {

		det = (double)det * a[i*dim+i];

	}
	
	return (double)(det);
}



__device__ void NewSource(Int64_t seed, randObj* rObj) {
	
	rng_Seed(seed, &(rObj->src));
	rObj->readVal = 0;
	rObj->readPos = 0;
	return;

}

__device__ void NewSource_EmeryFork(Int64_t seed, randObj* rObj) {
	rng_Seed_EmeryFork(seed, &(rObj->src));
	rObj->readVal = 0;
	rObj->readPos = 0;
	return ;
}



__device__ Int64_t rObj_Int63(randObj* r) {
	Int64_t x = rng_Int63(&(r->src));
	return x;
}

__device__ Int32_t rObj_Uint32(randObj* r) {
	Int32_t n = (rObj_Int63(r) >> 31);
	return n;

}


__device__ Int64_t rObj_Uint64(randObj* r) {
	if (&(r->src) != NULL) {
		return rng_Uint64(&(r->src));
	}
	Int64_t n = ((rObj_Int63(r)) >> 31 | (rObj_Int63(r)) << 32);
	return n;
}

__device__ int rObj_Int(randObj* r) {
	long long int u = (long long int)rObj_Int63(r);
	return (int)(u << 1) >> 1; // clear sign bit if int == int32
}

__device__ long int rObj_Int31(randObj* r)
{
	return (rObj_Int63(r) >> 32);

}



__device__ Int64_t rObj_Int63n(Int64_t n, randObj* r) {
	if (n <= 0) {
		printf("invalid argument to Int63n");
		return -1;

	}

	if ((n & (n - 1)) == 0) { // n is power of two, can mask

		return rObj_Int63(r) & (n - 1);
	}
	Uint64_t one = 1ULL;
	Int64_t max = (Int64_t)(one << 63) - one - (one << 63) % (Uint64_t)n;
	Int64_t v = rObj_Int63(r);


	while (v > max) {

		v = rObj_Int63(r);
	}

	return v % n;
}




__device__ Int64_t glb_seedrand32(Int64_t x) {

	Int64_t hi = x / Q;
	Uint64_t lo = x % Q;
	x = A * lo - R * hi;
	if (x < 0) {
		x += int32max;
	}
	return x;
}

__device__ Int64_t  glb_seedrand(Int64_t x) {

	Int64_t hi = x / Q1;
	Int64_t lo = x % Q1;
	x = A1 * lo - R1 * hi;
	if (x < 0) {
		x += int48max;
	}
	return x;
}


__device__ void rng_Seed(Int64_t seed, rngSource* rng) {
	rng->tap = 0;
	rng->feed = rngLen - rngTap;

	if (seed < 0) {
		seed += int64max;
	}
	if (seed == 0) {
		seed = 89482311;
	}

	Int64_t x = seed;
	for (int i = -30; i < rngLen; i++) {
		x = glb_seedrand(x);
		if (i >= 0) {
			Int64_t  u = 0;
			u = x << 40;
			x = glb_seedrand(x);
			u ^= x << 20;
			x = glb_seedrand(x);
			u ^= x;
			u ^= rngCooked[i];
			rng->vec[i] = u;
		}

	}
}

//generate rng->vec , an array used to generate a randomm matrix
__device__ void rng_Seed_EmeryFork(Int64_t seed, rngSource* rng) {
	rng->tap = 0;
	rng->feed = rngLen - rngTap;

	if (seed < 0) {
		seed += int64max;
	}
	if (seed == 0) {
		seed = 89482311;
	}

	Int64_t x = seed;
	for (int i = -30; i < rngLen; i++) {

		x = glb_seedrand(x);

		x = x ^ seed;

		if (i >= 0) {
			Int64_t u = 0LL;
			u = (Int64_t)(x << 40);
			x = glb_seedrand(x);
			x = x ^ seed;
			u = u ^ (x << 20);
			x = glb_seedrand(x);
			x = x ^ seed;
			u = u ^ x;
			u = u ^ rngCooked[i];
			rng->vec[i] = u;

		}
	}
}





__device__ Uint64_t  rng_Uint64(rngSource* rng) {

	rng->tap--;
	if (rng->tap < 0) {
		rng->tap += rngLen;
	}

	rng->feed--;
	if (rng->feed < 0) {
		rng->feed += rngLen;
	}

	Int64_t x = rng->vec[rng->feed] + rng->vec[rng->tap];

	rng->vec[rng->feed] = x;
	return (Uint64_t)x;
}

__device__ Int64_t rng_Int63(rngSource* rng) {
	Uint64_t x = rng_Uint64(rng);

	return (Int64_t)(x & rngMask);
}

//gpu core routine
__global__ void cuda_det(int* hashBytes, double* retDets, int* hashByteSize, int* mtrsize, int* height) {

	int hashsize = *hashByteSize;
    
	int msize = *mtrsize;
	int eHeight = (int)*height;
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x ;

	generateRandomMat_Det(hashBytes,retDets, hashsize, msize, eHeight,tid);
	
	return;

}



extern "C" {
	 void Determinant(int* hashBytes, double* retDets, int Blocks, int Threads, int mtrxSize,int hashBytesize,int Height) {
		 int blocks = Blocks;
		 int threads = Threads;
		 int mSize = mtrxSize; //now is set to 30
		 int hashSize = hashBytesize; // now is 32
		 int height = Height;
		int total = threads * blocks * hashBytesize;
		// Allocate device memory:
		int* gpu_hashbytes;
		double* gpu_retD;
		
		int* eheight;
		int* mtrSize;
		int* hashByteSize;
		
		cudaMalloc((int**)&gpu_hashbytes, sizeof(int) * total);
		cudaMalloc((double**)&gpu_retD, sizeof(double) * blocks* threads);
		cudaMalloc((int**)&eheight, sizeof(int));
		cudaMalloc((int**)&hashByteSize, sizeof(int));
		cudaMalloc((int**)&mtrSize, sizeof(int));
		cudaMemcpy((int*)gpu_hashbytes, hashBytes, sizeof(int)*total, cudaMemcpyHostToDevice);
		cudaMemcpy((int*)eheight, &height, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy((int*)mtrSize, &mSize, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy((int*)hashByteSize, &hashSize, sizeof(int), cudaMemcpyHostToDevice);
		cuda_det << <blocks, threads >> > (gpu_hashbytes, gpu_retD, hashByteSize,mtrSize, eheight);
		
		cudaMemcpy(retDets,  (double*)gpu_retD, sizeof(double) *blocks*threads, cudaMemcpyDeviceToHost);
		
		cudaFree(gpu_hashbytes);
		cudaFree(gpu_retD);
		cudaFree(hashByteSize);
		cudaFree(eheight);
		cudaFree(mtrSize);
		return;
	}
	
}
