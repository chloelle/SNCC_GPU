/*
Copyright (C) Chloe LeGendre

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TILE_WIDTH 32 

__global__ void SNCC_SummationFilter_Kernel_excl(float* device_costVol, float* device_costMap, unsigned char* device_disparityMap, int w, int h, int maxDisparity){
	
	int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int Col = blockIdx.x * TILE_WIDTH + threadIdx.x; 
	
	float A = 0;
	float B = 0;
	float C = 0;
	float D = 0;

	float corrScore = 0;
	float bestCorrScore = -RAND_MAX;
	int bestMatchSoFar = 0;
	
	if (Row > 5 && Col > 5 && Row < h - 7 && Col < w - 7){ //within summation filter bounds //was h-6, w-6 before
		for (int d = 0; d <= min(maxDisparity, Col - 6); d++){
			A = device_costVol[d*w*h + (Row - 6)*w + (Col - 6)]; // for exclusive scan. incl: [d*w*h + (Row - 7)*w + (Col - 7)];
			B = device_costVol[d*w*h + (Row - 6)*w + (Col + 7)]; // for exclusive scan. incl: [d*w*h + (Row - 7)*w + (Col + 6)];
			C = device_costVol[d*w*h + (Row + 7)*w + (Col - 6)]; // for exclusive scan. incl: [d*w*h + (Row + 6)*w + (Col - 7)];
			D = device_costVol[d*w*h + (Row + 7)*w + (Col + 7)]; // for exclusive scan. incl: [d*w*h + (Row + 6)*w + (Col + 6)];

			corrScore = (D - B - C + A)/169.0; //169 is for winAvg*winAvg

			if (bestCorrScore < corrScore){
					bestCorrScore = corrScore;
					bestMatchSoFar = d;
			}
		}

		device_costMap[Row*w + Col] = bestCorrScore;
		device_disparityMap[Row*w + Col] = unsigned char(bestMatchSoFar);
	}
}

__global__ void SNCC_DotProduct_KernelA(const float* __restrict__ device_patchMatrixLeft, const float* __restrict__ device_patchMatrixRight, float* __restrict__ device_costVol, int w, int h, int maxDisparity){

	int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int Col = blockIdx.x * TILE_WIDTH + threadIdx.x; 
	
	if (Row > 1 && Row < h - 2 && Col > 1 && Col < w - 2){ //within bounds for NCC window operation on image
	
		const int winsquared = 8;
		float leftTmp [winsquared];
		float NCC_val = 0.0;
		float rightTmp = 0.0;

		//load left vector
		for (int p = 0; p < winsquared; p++){
			leftTmp[p] = device_patchMatrixLeft[p*w*h + Row*w + Col];

		}

		//for all disparities
		for (int d = 0; d <= min(maxDisparity, Col - 2) ; d++){

			NCC_val = 0.0;
			for (int p = 0; p < winsquared; p++){
				rightTmp = device_patchMatrixRight[p*w*h + Row*w + Col - d];
				NCC_val += leftTmp[p]*rightTmp;
			}

			device_costVol[d*w*h + Row*w + Col] = NCC_val;
		}
	}
}

__global__ void SNCC_DotProduct_KernelB(const float* __restrict__ device_patchMatrixLeft, const float* __restrict__ device_patchMatrixRight, float* __restrict__ device_costVol, int w, int h, int maxDisparity){

	int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int Col = blockIdx.x * TILE_WIDTH + threadIdx.x; 
	
	if (Row > 1 && Row < h - 2 && Col > 1 && Col < w - 2){ //within bounds for NCC window operation on image
	
		const int winsquared = 16;
		float leftTmp [winsquared];
		float NCC_val = 0.0;
		float rightTmp = 0.0;

		//load left vector
		for (int p = 8; p < winsquared; p++){
			leftTmp[p] = device_patchMatrixLeft[p*w*h + Row*w + Col];

		}

		//for all disparities
		for (int d = 0; d <= min(maxDisparity, Col - 2) ; d++){ 

			NCC_val = 0.0;
			for (int p = 8; p < winsquared; p++){
				rightTmp = device_patchMatrixRight[p*w*h + Row*w + Col - d];
				NCC_val += leftTmp[p]*rightTmp;
			}

			device_costVol[d*w*h + Row*w + Col]+=NCC_val;
		}
	}
}

__global__ void SNCC_DotProduct_KernelC(const float* __restrict__ device_patchMatrixLeft, const float* __restrict__ device_patchMatrixRight, float* __restrict__ device_costVol, int w, int h, int maxDisparity){

	int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int Col = blockIdx.x * TILE_WIDTH + threadIdx.x; 
	
	if (Row > 1 && Row < h - 2 && Col > 1 && Col < w - 2){ //within bounds for NCC window operation on image
	
		const int winsquared = 25;
		float leftTmp [winsquared];
		float NCC_val = 0.0;
		float rightTmp = 0.0;

		//load left vector
		for (int p = 16; p < winsquared; p++){
			leftTmp[p] = device_patchMatrixLeft[p*w*h + Row*w + Col];

		}

		//for all disparities
		for (int d = 0; d <= min(maxDisparity, Col - 2) ; d++){ 

			NCC_val = 0.0;
			for (int p = 16; p < winsquared; p++){
				rightTmp = device_patchMatrixRight[p*w*h + Row*w + Col - d];
				NCC_val += leftTmp[p]*rightTmp;
			}

			device_costVol[d*w*h + Row*w + Col]+=NCC_val;
		}
	}
}

__global__ void SNCC_Mean_SD_Kernel(const unsigned char* __restrict__ device_left, const unsigned char* __restrict__ device_right, float* __restrict__ device_patchMatrixLeft, float* __restrict__ device_patchMatrixRight, int w, int h){

	int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int Col = blockIdx.x * TILE_WIDTH + threadIdx.x; 
	
	//create variable for shared memory
	__shared__ unsigned char tile_leftImage[TILE_WIDTH + 4][TILE_WIDTH + 4];
	__shared__ unsigned char tile_rightImage[TILE_WIDTH + 4][TILE_WIDTH + 4];

	//load left and right image data into shared memory
	if (Row < h && Col < w){
		tile_leftImage[threadIdx.y + 2][threadIdx.x + 2] = device_left[Row*w + Col];
		tile_rightImage[threadIdx.y + 2][threadIdx.x + 2] = device_right[Row*w + Col];
	}

	//load overlapping perimeter of block into shared memory (ghost elements)
	//LEFT pixels
	if (threadIdx.x == 0){
		if (Col > 1 && Row < h ){
			tile_leftImage[threadIdx.y + 2][threadIdx.x] = device_left[Row*w + Col - 2];
			tile_leftImage[threadIdx.y + 2][threadIdx.x + 1] = device_left[Row*w + Col - 1];
			tile_rightImage[threadIdx.y + 2][threadIdx.x] = device_right[Row*w + Col - 2];
			tile_rightImage[threadIdx.y + 2][threadIdx.x + 1] = device_right[Row*w + Col - 1];
		}
	}
	
	//RIGHT pixels
	if (threadIdx.x == TILE_WIDTH - 1){
		if (Col < (w - 2) && Row < h ){ 
			tile_leftImage[threadIdx.y + 2][threadIdx.x + 3] = device_left[Row*w + Col + 1];
			tile_leftImage[threadIdx.y + 2][threadIdx.x + 4] = device_left[Row*w + Col + 2];
			tile_rightImage[threadIdx.y + 2][threadIdx.x + 3] = device_right[Row*w + Col + 1];
			tile_rightImage[threadIdx.y + 2][threadIdx.x + 4] = device_right[Row*w + Col + 2];
		}
	}

	//TOP pixels
	if (threadIdx.y == 0){
		if (Row > 1 && Col < w ){
			tile_leftImage[threadIdx.y][threadIdx.x + 2] = device_left[(Row-2)*w + Col];
			tile_leftImage[threadIdx.y + 1][threadIdx.x + 2] = device_left[(Row-1)*w + Col];
			tile_rightImage[threadIdx.y][threadIdx.x + 2] = device_right[(Row-2)*w + Col];
			tile_rightImage[threadIdx.y + 1][threadIdx.x + 2] = device_right[(Row-1)*w + Col];
		}
	}

	//BOTTOM pixels
	if (threadIdx.y == TILE_WIDTH - 1){
		if (Row < h - 2 && Col < w ){
			tile_leftImage[threadIdx.y + 3][threadIdx.x + 2] = device_left[(Row+1)*w + Col];
			tile_leftImage[threadIdx.y + 4][threadIdx.x + 2] = device_left[(Row+2)*w + Col];
			tile_rightImage[threadIdx.y + 3][threadIdx.x + 2] = device_right[(Row+1)*w + Col];
			tile_rightImage[threadIdx.y + 4][threadIdx.x + 2] = device_right[(Row+2)*w + Col];
		}
	}
	
	//CORNERS
	//top left
	if (threadIdx.x == 0 && threadIdx.y == 0){
		if (Row > 1 && Col > 1){
			tile_leftImage[threadIdx.y][threadIdx.x] = device_left[(Row - 2)*w + Col - 2];
			tile_leftImage[threadIdx.y + 1][threadIdx.x] = device_left[(Row - 1)*w + Col - 2];
			tile_leftImage[threadIdx.y][threadIdx.x + 1] = device_left[(Row - 2)*w + Col - 1];
			tile_leftImage[threadIdx.y + 1][threadIdx.x + 1] = device_left[(Row - 1)*w + Col - 1];
			tile_rightImage[threadIdx.y][threadIdx.x] = device_right[(Row - 2)*w + Col - 2];
			tile_rightImage[threadIdx.y + 1][threadIdx.x] = device_right[(Row - 1)*w + Col - 2];
			tile_rightImage[threadIdx.y][threadIdx.x + 1] = device_right[(Row - 2)*w + Col - 1];
			tile_rightImage[threadIdx.y + 1][threadIdx.x + 1] = device_right[(Row - 1)*w + Col - 1];
		}
	}
	
	//top right
	if (threadIdx.x == TILE_WIDTH - 1 && threadIdx.y == 0){
		if (Row > 1 && Col < w - 2){
			tile_leftImage[threadIdx.y][threadIdx.x + 4] = device_left[(Row - 2)*w + Col + 2];
			tile_leftImage[threadIdx.y][threadIdx.x + 3] = device_left[(Row - 2)*w + Col + 1];
			tile_leftImage[threadIdx.y + 1][threadIdx.x + 3] = device_left[(Row - 1)*w + Col + 1];
			tile_leftImage[threadIdx.y + 1][threadIdx.x + 4] = device_left[(Row - 1)*w + Col + 2];
			tile_rightImage[threadIdx.y][threadIdx.x + 4] = device_right[(Row - 2)*w + Col + 2];
			tile_rightImage[threadIdx.y][threadIdx.x + 3] = device_right[(Row - 2)*w + Col + 1];
			tile_rightImage[threadIdx.y + 1][threadIdx.x + 3] = device_right[(Row - 1)*w + Col + 1];
			tile_rightImage[threadIdx.y + 1][threadIdx.x + 4] = device_right[(Row - 1)*w + Col + 2];
		}
	}
	
	//bottom left
	if (threadIdx.x == 0 && threadIdx.y == TILE_WIDTH - 1){
		if (Row < h - 2 && Col > 1){
			tile_leftImage[threadIdx.y + 4][threadIdx.x] = device_left[(Row + 2)*w + Col - 2];
			tile_leftImage[threadIdx.y + 3][threadIdx.x] = device_left[(Row + 1)*w + Col - 2];
			tile_leftImage[threadIdx.y + 4][threadIdx.x + 1] = device_left[(Row + 2)*w + Col - 1];
			tile_leftImage[threadIdx.y + 3][threadIdx.x + 1] = device_left[(Row + 1)*w + Col - 1];
			tile_rightImage[threadIdx.y + 4][threadIdx.x] = device_right[(Row + 2)*w + Col - 2];
			tile_rightImage[threadIdx.y + 3][threadIdx.x] = device_right[(Row + 1)*w + Col - 2];
			tile_rightImage[threadIdx.y + 4][threadIdx.x + 1] = device_right[(Row + 2)*w + Col - 1];
			tile_rightImage[threadIdx.y + 3][threadIdx.x + 1] = device_right[(Row + 1)*w + Col - 1];
			
		}
	}

	//bottom right
	if (threadIdx.x == TILE_WIDTH -1 && threadIdx.y == TILE_WIDTH - 1){
		if (Row < h - 2 && Col < w - 2){
			tile_leftImage[threadIdx.y + 4][threadIdx.x + 4] = device_left[(Row + 2)*w + Col + 2];
			tile_leftImage[threadIdx.y + 3][threadIdx.x + 4] = device_left[(Row + 1)*w + Col + 2];
			tile_leftImage[threadIdx.y + 3][threadIdx.x + 3] = device_left[(Row + 1)*w + Col + 1];
			tile_leftImage[threadIdx.y + 4][threadIdx.x + 3] = device_left[(Row + 2)*w + Col + 1];
			tile_rightImage[threadIdx.y + 4][threadIdx.x + 4] = device_right[(Row + 2)*w + Col + 2];
			tile_rightImage[threadIdx.y + 3][threadIdx.x + 4] = device_right[(Row + 1)*w + Col + 2];
			tile_rightImage[threadIdx.y + 3][threadIdx.x + 3] = device_right[(Row + 1)*w + Col + 1];
			tile_rightImage[threadIdx.y + 4][threadIdx.x + 3] = device_right[(Row + 2)*w + Col + 1];
		}
	}

	__syncthreads();

	//set boundaries to 0 in patchMatrixLeft and patchMatrixRight
	if ((Row == 0 || Row == 1) && Col < w ){ //top
		for (int d = 0; d < 25; d++){
			device_patchMatrixLeft[d*w*h + Row*w + Col] = 0;
			device_patchMatrixRight[d*w*h + Row*w + Col] = 0;
		}
	}
	else if ((Row == h - 1 || Row == h - 2) && Col < w){ //bottom
		for (int d = 0; d < 25; d++){
			device_patchMatrixLeft[d*w*h + Row*w + Col] = 0;
			device_patchMatrixRight[d*w*h + Row*w + Col] = 0;
		}
	}
	else if ((Col == 0 || Col == 1) && Row < h){//left
		for (int d = 0; d < 25; d++){
			device_patchMatrixLeft[d*w*h + Row*w + Col] = 0;
			device_patchMatrixRight[d*w*h + Row*w + Col] = 0;
		}	
	}
	else if ((Col == w - 1 || Col == w - 2) && Row < h){//right
		for (int d = 0; d < 25; d++){
			device_patchMatrixLeft[d*w*h + Row*w + Col] = 0;
			device_patchMatrixRight[d*w*h + Row*w + Col] = 0;
		}	
	}
	else {}

	//compute mean of each patch
	float leftMean = 0; 
	float rightMean = 0;
	float leftSD = 0; 
	float rightSD = 0;
	float winsquared = 25.0;
	int idx = 0;

	if ( Row < h - 2 && Row > 1 && Col < w - 2 && Col > 1){
		//iterate over window to get sum
		for (int p = threadIdx.y; p <= threadIdx.y + 4; p++ ){
			for (int q = threadIdx.x; q <= threadIdx.x + 4; q++){
				leftMean += float(tile_leftImage[p][q]);
				rightMean += float(tile_rightImage[p][q]);
			}
		}

		leftMean = leftMean/(winsquared);
		rightMean = rightMean/(winsquared);

		//iterate over window to get SD
		for (int p = threadIdx.y; p <= threadIdx.y + 4; p++ ){
			for (int q = threadIdx.x; q <= threadIdx.x + 4; q++){
				leftSD += pow((float(tile_leftImage[p][q]) - leftMean), 2);
				rightSD += pow((float(tile_rightImage[p][q]) - rightMean), 2);
			}
		}

		//iterate over window to subtract mean, divide by sqrt of sd
		for (int p = threadIdx.y; p <= threadIdx.y + 4; p++ ){
			for (int q = threadIdx.x; q <= threadIdx.x + 4; q++){
				if (leftSD == 0){
					device_patchMatrixLeft[idx*w*h + Row*w + Col] = 0;
				}
				else{
					device_patchMatrixLeft[idx*w*h + Row*w + Col] = ( float(tile_leftImage[p][q])  - leftMean)/sqrt(leftSD);
				}
				if (rightSD == 0){
					device_patchMatrixRight[idx*w*h + Row*w + Col] = 0;
				}
				else{
					device_patchMatrixRight[idx*w*h + Row*w + Col] = ( float(tile_rightImage[p][q]) - rightMean)/sqrt(rightSD);
				}
				idx++;
			}
		}
	}
}

__global__ void exclusive_scan_volume_kernelRow(const float* __restrict__ device_costVol, float* __restrict__ device_costVolIntegral, int w, int h){
	
	__shared__ float temp[2048];
	int tdx = threadIdx.x; 
	int offset = 1;
	int n = 2048;

	if (2*tdx+1 < w){
		temp[2*tdx] = device_costVol[ blockIdx.z*w*h + blockIdx.y*w + 2*tdx];
		temp[2*tdx+1] = device_costVol[ blockIdx.z*w*h + blockIdx.y*w + 2*tdx+1];
	}
	else if(2*tdx < w){
		temp[2*tdx] = device_costVol[ blockIdx.z*w*h + blockIdx.y*w + 2*tdx];
		temp[2*tdx+1] = 0;
	}
	else{
		temp[2*tdx] = 0;
		temp[2*tdx+1] = 0;
	}

	for(int d = n>>1; d > 0; d >>= 1){
		__syncthreads();
		if(tdx < d){
			int ai = offset*(2*tdx+1)-1;
			int bi = offset*(2*tdx+2)-1;
			temp[bi] += temp[ai];
		}
		offset <<= 1; //offset *= 2;
	}

	if(tdx == 0) temp[n - 1] = 0;
	for(int d = 1; d < n; d <<= 1){ //d *= 2
		offset >>= 1; __syncthreads();
		if(tdx < d){
			int ai = offset*(2*tdx+1)-1;
			int bi = offset*(2*tdx+2)-1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	if (2*tdx+1 < w){
		device_costVolIntegral[ blockIdx.z*w*h + 2*tdx*h + blockIdx.y] = temp[2*tdx];
		device_costVolIntegral[ blockIdx.z*w*h + (2*tdx+1)*h + blockIdx.y] = temp[2*tdx+1];
	}
	else if (2*tdx < w){
		device_costVolIntegral[ blockIdx.z*w*h + 2*tdx*h + blockIdx.y] = temp[2*tdx];
	}
	else {} //do nothing
}

__global__ void exclusive_scan_volume_kernelCol(float* __restrict__ device_costVol, const float* __restrict__ device_costVolIntegral, int w, int h){
	
	__shared__ float temp[512];
	int tdx = threadIdx.x; 
	int offset = 1;
	int n = 512;

	if (2*tdx+1 < h){
		temp[2*tdx] = device_costVolIntegral[ blockIdx.z*w*h + blockIdx.x*h + 2*tdx ];
		temp[2*tdx+1] = device_costVolIntegral[ blockIdx.z*w*h + blockIdx.x*h + (2*tdx+1) ];

	}
	else if(2*tdx < h){
		temp[2*tdx] = device_costVolIntegral[ blockIdx.z*w*h + blockIdx.x*h + 2*tdx ];
		temp[2*tdx+1] = 0;
	}
	else{
		temp[2*tdx] = 0;
		temp[2*tdx+1] = 0;
	}

	for(int d = n>>1; d > 0; d >>= 1){
		__syncthreads();
		if(tdx < d){
			int ai = offset*(2*tdx+1)-1;
			int bi = offset*(2*tdx+2)-1;
			temp[bi] += temp[ai];
		}
		offset <<= 1; //offset *= 2;
	}

	if(tdx == 0) temp[n - 1] = 0;
	for(int d = 1; d < n; d <<= 1){ //d *= 2
		offset >>= 1; __syncthreads();
		if(tdx < d){
			int ai = offset*(2*tdx+1)-1;
			int bi = offset*(2*tdx+2)-1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	if (2*tdx+1 < h){
		device_costVol[ blockIdx.z*w*h + 2*tdx*w + blockIdx.x] = temp[2*tdx];
		device_costVol[ blockIdx.z*w*h + (2*tdx+1)*w + blockIdx.x] = temp[2*tdx+1];
	}
	else if (2*tdx < h){
		device_costVol[ blockIdx.z*w*h + 2*tdx*w + blockIdx.x] = temp[2*tdx];
	}
	else {} //do nothing
}

int SNCC_Stereo_Matching_GPU(unsigned char *left, unsigned char *right, unsigned char *disparityImage, float *costMap, int w, int h, int win, int winAvg, int maxDisparity){
	
	std::cout << "in GPU SNCC function. Image size:" << w << " " << h << std::endl;
	
	if (win != 5 || winAvg != 13){
		std::cout << "Window size error. To use GPU, NCC window size must by 5x5 and summation window size must be 13x13."  << std::endl;
		cudaDeviceReset();
		return 0;
	}

	//make sure we have a GPU
	int nDevices;
	cudaError_t err = cudaGetDeviceCount(&nDevices);
	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

	//test GPU memory allocation for left and right stereo images
	int numBytesImage = w*h*sizeof(unsigned char);
	unsigned char *device_left = 0;
	unsigned char *device_right = 0;
	err = cudaMalloc((void**)&device_left, numBytesImage);
	err = cudaMalloc((void**)&device_right, numBytesImage);
	if (device_left == 0 || device_right == 0){
		printf("couldn't allocate memory for images \n");
		cudaFree(device_left);
		cudaFree(device_right);
	}
	if( err != cudaSuccess)
    printf("cudaMalloc error: %s\n", cudaGetErrorString(err));

	//copy images to device
	err = cudaMemcpy(device_left, left, numBytesImage, cudaMemcpyHostToDevice);
	err = cudaMemcpy(device_right, right, numBytesImage, cudaMemcpyHostToDevice);
	if( err != cudaSuccess)
    printf("cudaMemcpy error: %s\n", cudaGetErrorString(err));

	//test GPU memory allocation for patch matrices
	int numBytesPatch = w*h*(25)*sizeof(float); //25 is win*win
	float *device_patchMatrixLeft = 0; 
	float *device_patchMatrixRight = 0;
	err = cudaMalloc((void**)&device_patchMatrixLeft, numBytesPatch);
	err = cudaMalloc((void**)&device_patchMatrixRight, numBytesPatch);
	if (device_patchMatrixLeft == 0 || device_patchMatrixRight == 0){
		printf("couldn't allocate memory for patch matrices\n");
		cudaFree(device_patchMatrixLeft);
		cudaFree(device_patchMatrixRight);
		return 0;
	}
	if( err != cudaSuccess)
    printf("cudaMalloc error: %s\n", cudaGetErrorString(err));

	//test GPU memory allocation for cost volume	
	int numBytesCostVol = w*h*(maxDisparity+1)*sizeof(float);
	float *device_costVol = 0;
	err = cudaMalloc((void**)&device_costVol, numBytesCostVol);
	if (device_costVol == 0){
		printf("couldn't allocate memory for cost volume\n");
		cudaFree(device_costVol);
		return 0;
	}
	if( err != cudaSuccess)
    printf("cudaMalloc error: %s\n", cudaGetErrorString(err));

	//test GPU memory allocation for cost volume integral image
	float *device_costVolIntegral = 0;
	err = cudaMalloc((void**)&device_costVolIntegral, numBytesCostVol);
	if (device_costVolIntegral == 0){
		printf("couldn't allocate memory for cost volume integral\n");
		cudaFree(device_costVolIntegral);
		return 0;
	}
	if( err != cudaSuccess)
    printf("cudaMalloc error: %s\n", cudaGetErrorString(err));
	
	//test GPU memory allocation for cost map
	int numBytesCostMap = w*h*sizeof(float);
	float *device_costMap = 0; 
	err = cudaMalloc((void**)&device_costMap, numBytesCostMap);
	if (device_costMap == 0){
		printf("couldn't allocate memory for cost map \n");
		cudaFree(device_costMap);
		return 0;
	}
	if( err != cudaSuccess)
    printf("cudaMalloc error: %s\n", cudaGetErrorString(err));

	//test GPU memory allocation for disparity map
	int numBytesDisparityMap = w*h*sizeof(unsigned char);
	unsigned char *device_disparityMap = 0;
	err = cudaMalloc((void**)&device_disparityMap, numBytesDisparityMap);
	if (device_disparityMap == 0){
		printf("coudln't allocate memory for disparity map\n");
		cudaFree(device_disparityMap);
		return 0;
	}
	if( err != cudaSuccess)
    printf("cudaMalloc error: %s\n", cudaGetErrorString(err));

	//setup execution configuration for first kernel: mean and SD kernel
	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);
	int gridx = int(ceil(float(w)/float(TILE_WIDTH)));
	int gridy = int(ceil(float(h)/float(TILE_WIDTH)));
	dim3 dimGrid(gridx, gridy);
	const unsigned int shared_mem_size = sizeof(unsigned char)*((4+TILE_WIDTH)*(4+TILE_WIDTH))*2; //*2 for left and right image

	//launch first kernel: mean and SD kernel
	SNCC_Mean_SD_Kernel<<<dimGrid, dimBlock, shared_mem_size>>>(device_left, device_right, device_patchMatrixLeft, device_patchMatrixRight, w, h);

	/*
	//uncomment this part for error checking kernel execution
	err=cudaDeviceSynchronize();
	if( err != cudaSuccess)
    printf("cudaDeviceSynchronize error on Mean SD kernel: %s\n", cudaGetErrorString(err));
	*/
	std::cout << "finished kernel to pre-compute vector mean and SD" << std::endl;
	
	//setup execution configuration for second kernel
	//preset all costVol bytes to 0
	err = cudaMemset( device_costVol, 0,  numBytesCostVol);
	if( err != cudaSuccess)
    printf("error on cudaMemset 0: %s\n", cudaGetErrorString(err));

	//launch second kernel: compute NCC volume over all possible disparity ranges
	SNCC_DotProduct_KernelA<<<dimGrid, dimBlock>>>(device_patchMatrixLeft, device_patchMatrixRight, device_costVol, w, h, maxDisparity);
	
	/*
	//uncomment this part for error checking kernel execution
	err=cudaDeviceSynchronize();
	if( err != cudaSuccess)
    printf("cudaDeviceSynchronize error on Dot Product kernel A: %s\n", cudaGetErrorString(err));
	*/
	std::cout << "finished kernel dot product A" << std::endl;
	
	SNCC_DotProduct_KernelB<<<dimGrid, dimBlock>>>(device_patchMatrixLeft, device_patchMatrixRight, device_costVol, w, h, maxDisparity);
	
	/*
	//uncomment this part for error checking kernel execution
	err=cudaDeviceSynchronize();
	if( err != cudaSuccess)
    printf("cudaDeviceSynchronize error on Dot Product kernel B: %s\n", cudaGetErrorString(err));
	*/
	std::cout << "finished kernel dot product B" << std::endl;
	
	SNCC_DotProduct_KernelC<<<dimGrid, dimBlock>>>(device_patchMatrixLeft, device_patchMatrixRight, device_costVol, w, h, maxDisparity);
	
	/*
	//uncomment this part for error checking kernel execution
	err=cudaDeviceSynchronize();
	if( err != cudaSuccess)
    printf("cudaDeviceSynchronize error on Dot Product kernel C: %s\n", cudaGetErrorString(err));
	*/
	std::cout << "finished kernel dot product C" << std::endl;

	//free patch matrix memory
	err = cudaFree(device_patchMatrixLeft);
	err = cudaFree(device_patchMatrixRight);
	if( err != cudaSuccess)
    printf("cudaFree error: %s\n", cudaGetErrorString(err));

	//compute integral image on GPU, max width is 2048
	dim3 dimBlock2(1024,1,1);
	dim3 dimGrid2(1, h, maxDisparity);
	int shared_mem_size2 = 2048*sizeof(float);
	exclusive_scan_volume_kernelRow<<< dimGrid2, dimBlock2, shared_mem_size2>>>(device_costVol, device_costVolIntegral, w, h);
	
	/*
	//uncomment this part for error checking kernel execution
	err=cudaDeviceSynchronize();
	if( err != cudaSuccess)
    printf("cudaDeviceSynchronize error on exclusive_scan_volume_kernelRow: %s\n", cudaGetErrorString(err));
	*/
	std::cout << "finished exclusive scan kernel across rows" << std::endl;

	dim3 dimBlock3(256,1,1);
	dim3 dimGrid3(w, 1, maxDisparity);
	int shared_mem_size3 = 512*sizeof(float);
	exclusive_scan_volume_kernelCol<<< dimGrid3, dimBlock3, shared_mem_size3>>>(device_costVol, device_costVolIntegral, w, h);
	
	/*
	//uncomment this part for error checking kernel execution
	err=cudaDeviceSynchronize();
	if( err != cudaSuccess)
    printf("cudaDeviceSynchronize error on exclusive_scan_volume_kernelCol: %s\n", cudaGetErrorString(err));
	*/
	std::cout << "finished exclusive scan kernel across columns" << std::endl;
	
	//preset disparity map bytes to 0
	err = cudaMemset( device_disparityMap, 0, numBytesDisparityMap);
	if( err != cudaSuccess)
    printf("error on cudaMemset 0: %s\n", cudaGetErrorString(err));

	//launch third kernel: apply summation filter using integral image, based on exclusive scan
	SNCC_SummationFilter_Kernel_excl<<<dimGrid, dimBlock>>>(device_costVol, device_costMap, device_disparityMap, w, h, maxDisparity);
	
	/*
	//uncomment this part for error checking kernel execution
	err=cudaDeviceSynchronize();
	if( err != cudaSuccess)
    printf("cudaDeviceSynchronize error on SNCC_SummationFilter_Kernel_excl: %s\n", cudaGetErrorString(err));
	*/
	std::cout << "finished summation kernel" << std::endl;
	
	//copy device cost map and disparity map to host
	err = cudaMemcpy(costMap, device_costMap, numBytesCostMap, cudaMemcpyDeviceToHost);
	err = cudaMemcpy(disparityImage, device_disparityMap, numBytesDisparityMap, cudaMemcpyDeviceToHost); 
	if( err != cudaSuccess)
    printf("cudaMemcpy error: %s\n", cudaGetErrorString(err));
	
	std::cout << "saving disparity map and cost map" << std::endl;

	cudaFree(device_costVolIntegral);
	cudaFree(device_costVol);
	cudaFree(device_costMap);
	cudaFree(device_disparityMap);
	cudaFree(device_left);
	cudaFree(device_right);
	
	cudaDeviceReset();
	return 0;
}