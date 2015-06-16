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

//added by CL for visual studio 2013 
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "ImageFunctions.h"
#include <math.h>
#include <climits>
#include <cfloat>
#include <cstring>
#include <algorithm>

#include <cuda_runtime.h>

//Declaration forward: compute disparity map with SNCC on the GPU.
int SNCC_Stereo_Matching_GPU(unsigned char *left, unsigned char *right, unsigned char *disparityImage, float *costMap, int w, int h, int win, int winAvg, int maxDisparity);

using namespace std;

//Print array function to test binary I/O
void printArrayFloat(float** array, int w, int h){
	printf("Array: \n");
	for (int j = 0; j < h; j++){
		for (int i = 0; i < w; i++){
			printf("%6f \t", array[j][i]);
		}
		printf("\n");
	}
}

//Read data from file for cost map (binary). Not used but included.
void readBinaryFloatArray(float *vector, char arrayFileName[], int w, int h, int depth){
	//read data from file for cost map
	ifstream fin;
	fin.open(arrayFileName, ios::binary);
	if (fin.fail()){
		cout << "File I/O error" << endl;
		exit(0);
	}
	int vectorSize = (w*h*(depth + 1));
	binaryFileToArrayFloat(fin, vector, vectorSize);
	fin.close();
}

//Write data to file for cost map (binary).
void writeBinaryFloatArray(float *vector, char arrayFileName[], int w, int h, int depth){
	ofstream fout;
	fout.open(arrayFileName, ios::binary);
	if (fout.fail()) {
		cout << "File I/O error" << endl;
	}
	int vectorSize = (w*h*(depth + 1));
	arrayFloatToBinaryFile(vector, vectorSize, fout);
	fout.close();
}

//Compute disparity map with SNCC on the CPU.
void SNCC_Stereo_Matching(unsigned char *left, unsigned char *right, unsigned char *disparityImage, float *costMap, int w, int h, int win, int winAvg, int maxDisparity){

	if (win%2 != 1){ //check for odd win size
		win--;
	}
	int win2 = int(floor(win/2.0));

	if (winAvg%2 != 1){ //check for odd win size
		winAvg--;
	}
	int winAvg2 = int(floor(winAvg/2.0));

	//Initialize correlation score matrix as 0's
	float* NCC_map = new float[w*h*(maxDisparity+1)];
	for (int i = 0; i < (w*h*(maxDisparity+1)); i++){
		NCC_map[i] = 0;
	}

	//start costMap as min
	for (int j = 0; j < h; j++){
		for (int i = 0; i < w; i++){
			costMap[w*j + i] = float(-RAND_MAX);
		}
	}

	//PRECOMPUTE mean-adjusted, stdev-scaled patch vectors
	float ***leftPatchMatrix;
	leftPatchMatrix = new float** [h];
	float ***rightPatchMatrix;
	rightPatchMatrix = new float** [h];

	for (int j = 0; j < h; j++){
		leftPatchMatrix[j] = new float*[w];
		rightPatchMatrix[j] = new float*[w];
	}
	for (int j = 0; j < h; j++){
		for (int i = 0; i < w; i++){
			leftPatchMatrix[j][i] = new float[win*win];
			rightPatchMatrix[j][i] = new float[win*win];
		}
	}

	//iterate over all pixels to populate matrices
	float* leftTmp;
	float* rightTmp;
	leftTmp = new float [win*win];
	rightTmp = new float [win*win];
	float meanLeft = 0.00;
	float meanRight = 0.00;
	int count = 0;
	float sdLeft = 0.00;
	float sdRight = 0.00;

	for (int j = win2; j < h - win2; j++){
		for (int i = win2; i < w - win2; i++){

			//get window of pixel intensity values into vector
			count = 0;
			for(int p = j - win2; p <= j + win2; p++ ){
				for (int q = i - win2; q <= i + win2; q++){
					leftTmp[count] = float(left[p*w + q]);
					rightTmp[count] = float(right[p*w + q]);
					count++;
				}
			}

			//iterate over vector to get mean
			meanLeft = 0.00;
			meanRight = 0.00;
			for (int p = 0; p < win*win; p++){
				meanLeft += leftTmp[p];
				meanRight += rightTmp[p];
			}
			meanLeft = meanLeft/float(win*win);
			meanRight = meanRight/float(win*win);

			//iterate over vector to get SD
			sdLeft = 0.00;
			sdRight = 0.00;
			for (int p = 0; p < win*win; p++){
				sdLeft += pow((leftTmp[p] - meanLeft),2);
				sdRight += pow((rightTmp[p] - meanRight),2);
			}

			//iterate over vector to subtract mean and divide by SD
			for (int p = 0; p < win*win; p++){
				leftTmp[p] = (leftTmp[p] - meanLeft)/sqrt(sdLeft);
				rightTmp[p] = (rightTmp[p] - meanRight)/sqrt(sdRight);
				leftPatchMatrix[j][i][p] = leftTmp[p];
				rightPatchMatrix[j][i][p] = rightTmp[p];
			}
		}
	}

	//build NCC correlation volume
	float NCC_val = 0.00;
	for (int j = win2; j < h - win2; j++){
		for (int i = win2; i < w - win2; i++){

			leftTmp = leftPatchMatrix[j][i];

			for (int d = 0; d <= min(maxDisparity, i - win2); d++){

				rightTmp = rightPatchMatrix[j][i - d];

				//compute NCC
				NCC_val = 0.00;
				for (int p = 0; p < win*win; p++){
					NCC_val += leftTmp[p]*rightTmp[p]; //dot product
				}
				NCC_map[d*w*h + j*w + i] = NCC_val;
			}
		}
	}

	//delete vector matrices
	for (int j = 0; j < h; j++){
		for (int i = 0; i < w; i++){
			delete[] leftPatchMatrix[j][i];
			delete[] rightPatchMatrix[j][i];
		}
		delete[] leftPatchMatrix[j];
		delete[] rightPatchMatrix[j];
	}
	delete[] leftPatchMatrix;
	delete[] rightPatchMatrix;

	//summation filtering (mean filter winAvg x winAvg size)
	float sumNCC;
	float bestCorrScore;
	float corrScore;
	int bestMatchSoFar;
	for (int j = winAvg2; j < h - winAvg2; j++){
		for (int i = winAvg2; i < w - winAvg2; i++){
			bestCorrScore = -RAND_MAX;
			bestMatchSoFar = 0;

			for (int d = 0; d <= min(maxDisparity, i - winAvg2); d++){
				//get mean
				sumNCC = 0.00;
				for (int r = j - winAvg2; r <= j + winAvg2; r++){
					for (int c = i - winAvg2; c <= i + winAvg2; c++){
						if (NCC_map[d*w*h + r*w + c] > -RAND_MAX){
							sumNCC+= NCC_map[d*w*h + r*w + c];
						}
					}
				}
				corrScore = sumNCC/float(winAvg*winAvg);

				if (bestCorrScore < corrScore){
					bestCorrScore = corrScore;
					bestMatchSoFar = d;
				}
			}
			disparityImage[w*j + i] = bestMatchSoFar;
			costMap[w*j + i] = float(bestCorrScore);
		}
	}
	delete[] NCC_map;
}

//Helper function to read the stereo pair images, call the GPU disparity map computation method, and save the disparity and cost maps to file.
void generateDisparityMapGPU(char leftFileName[], char rightFileName[], char dispMapSaveName[], char costMapSaveName[], int w, int h, int winSize, int avgwinSize, int maxDisparity){

	//create left image 1D array
	unsigned char* leftimg;
	leftimg = read_pgm_image(leftFileName, &w, &h);
	cout << "Left image: width x height:" << endl;
	cout << w << " " << h << endl;

	//create right image 1D array
	unsigned char* rightimg;
	rightimg = read_pgm_image(rightFileName, &w, &h);
	cout << "Right image: width x height:" << endl;
	cout << w << " " << h << endl;

	//create disparity map 1D array
	unsigned char *disparityMap;
	disparityMap = new unsigned char [w*h];

	//create cost map
	//cost map values need to be set in stereo function
	float *costMap;
	costMap = new float[w*h];
	
	SNCC_Stereo_Matching_GPU(leftimg, rightimg, disparityMap, costMap, w, h, winSize, avgwinSize, maxDisparity);
	cout << "producing result with SNCC on GPU" << endl;

	write_pgm_Uimage(disparityMap, dispMapSaveName, w, h); 
	writeBinaryFloatArray(costMap, costMapSaveName, w, h, 0);

	//cleanup arrays
	delete[] leftimg;
	delete[] rightimg;
	delete[] disparityMap;
	delete[] costMap;

}

//Helper function to read the stereo pair images, call the CPU disparity map computation method, and save the disparity and cost maps to file.
void generateDisparityMap(char leftFileName[], char rightFileName[], char dispMapSaveName[], char costMapSaveName[], int w, int h, int winSize, int avgwinSize, int maxDisparity){

	//create left image 1D array
	unsigned char* leftimg;
	leftimg = read_pgm_image(leftFileName, &w, &h);
	cout << "Left image: width x height:" << endl;
	cout << w << " " << h << endl;

	//create right image 1D array
	unsigned char* rightimg;
	rightimg = read_pgm_image(rightFileName, &w, &h);
	cout << "Right image: width x height:" << endl;
	cout << w << " " << h << endl;

	//create disparity map 1D array and set to 0
	unsigned char *disparityMap;
	disparityMap = new unsigned char[w*h];
	for (int j = 0; j < h; j++){ //row
		for (int i = 0; i < w; i++){ //col
			disparityMap[w*j + i] = 0;
		}
	}

	//create cost map
	//cost map values need to be set in stereo function
	float *costMap;
	costMap = new float[w*h];

	SNCC_Stereo_Matching(leftimg, rightimg, disparityMap, costMap, w, h, winSize, avgwinSize, maxDisparity);
	cout << "producing result with SNCC" << endl;

	write_pgm_Uimage(disparityMap, dispMapSaveName, w, h);
	writeBinaryFloatArray(costMap, costMapSaveName, w, h, 0);
	
	//cleanup arrays
	delete[] leftimg;
	delete[] rightimg;
	delete[] disparityMap;
	delete[] costMap;
}

/* 
to run need the following in root directory:
cudaSNCC.exe placed in root folder
output directories: ./disparity_GPU_SNCC5x5_13x13; ./disparity_CPU_SNCC5x5_13x13; ./costmap_GPU_SNCC5x5_13x13; ./costmap_CPU_SNCC5x5_13x13; 
for CPU, NCC window size and summation filter window size can be modified in main below.
output directories should be changed to reflect window size.
for GPU, NCC window size fixed at 5 and summation filter window size fixed at 13.
input directories: ./training_image_0; ./training_image_1
left images of stereo pair in folder ./training_image_0/
right images of stereo pair in folder ./training_image_1/
*/
int main(int argc, char *argv[] ) {

	//if true, code will use GPU to compute disparity map; if false, code will use CPU
	bool useGPU = true; 

	char leftFileName[100];
	char rightFileName[100];
	char dispMapSaveName[100];
	char costMapSaveName[100];

	//initialize w and h, these will be set correctly later when image files are read
	int w = 0;
	int h = 0;

	/*
	KITTI image data naming convention: 000AAA_BB.pgm where AAA is sequence ID and BB is frame ID
	total number of sequences in folder = 194 training (idx 0:193) and 195 testing (idx 0:194)
	total number of frames per sequence = 21 (idx 0:20)
	*/

	//can edit below to control which sequences and frames have disparity maps generated
	int numSeqMin = 0; //starting sequence ID: numSeqMin >= 0 && numSeqMin <= 193/194
	int numSeqMax = 5; //ending sequence ID: numSeqMax >= 0 && numSeqMax <= 193/194, && numSeqMax >= numSeqMin
	int numFramesMin = 8; //starting frame ID: numFramesMin >= 0 && numFramesMin <= 20
	int numFramesMax = 10; //ending frame ID: numFramesMax >= 0 && numFramesMax <= 20, && numFramesMax >= numFramesMin
	double baseline = 0.54; //stereo baseline
	int maxDisparity = 255;

	//for CPU code, these can be modified. for GPU, winSize = 5 && avgwinSize = 13 always, or will display error message.
	int winSize = 5; //NCC window size
	int avgwinSize = 13; //summation filter window size
	char stereoMethod[] = "SNCC";
	
	//uncomment to set NCC window size and summation filter window size as input arguments to program
	//int winSize = atoi(argv[1]); //first arg is winSize
	//int avgwinSize = atoi(argv[2]); //second arg is avgwinSize for SNCC

	for (int numSeq = numSeqMin; numSeq <= numSeqMax; numSeq++){
		//if (numSeq == 127 || numSeq == 182){ //missing data TESTING KITTI - uncomment this line and comment next line if using testing data
		if (numSeq == 31 || numSeq == 82 || numSeq == 114){ //missing data TRAINING KITTI
			continue;
		}
		else{
			for(int numFrames = numFramesMin; numFrames <= numFramesMax; numFrames ++){

				sprintf(leftFileName, "./training_image_0/%06d_%02d.pgm", numSeq, numFrames); 
				sprintf(rightFileName, "./training_image_1/%06d_%02d.pgm", numSeq, numFrames);

				cout << "Left image name: " << leftFileName << endl;
				cout << "Right image name: " << rightFileName << endl;

				if (useGPU){
					sprintf(costMapSaveName, "./costmap_GPU_%s%dx%d_%dx%d/%06d_%02d.bin", stereoMethod, winSize, winSize, avgwinSize, avgwinSize, numSeq, numFrames);
					sprintf(dispMapSaveName, "./disparity_GPU_%s%dx%d_%dx%d/%06d_%02d.pgm", stereoMethod, winSize, winSize, avgwinSize, avgwinSize, numSeq, numFrames);
					generateDisparityMapGPU(leftFileName, rightFileName, dispMapSaveName, costMapSaveName, w, h, winSize, avgwinSize, maxDisparity);
				}
				else{
					sprintf(costMapSaveName, "./costmap_CPU_%s%dx%d_%dx%d/%06d_%02d.bin", stereoMethod, winSize, winSize, avgwinSize, avgwinSize, numSeq, numFrames);
					sprintf(dispMapSaveName, "./disparity_CPU_%s%dx%d_%dx%d/%06d_%02d.pgm", stereoMethod, winSize, winSize, avgwinSize, avgwinSize, numSeq, numFrames);
					generateDisparityMap(leftFileName, rightFileName, dispMapSaveName, costMapSaveName, w, h, winSize, avgwinSize, maxDisparity);
				}
			}
		}
	}
	return 0;
}
