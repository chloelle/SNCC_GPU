# SNCC_GPU
compute disparity map from stereo pair using SNCC matching function (GPU and CPU code provided)

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

Title: SNCC stereo matching for KITTI data
Author: Chloe LeGendre, chloelle (at) gmail.com
Date: June 16, 2015

Computes disparity maps for KITTI data using SNCC matching function, per the following reference:
N. Einecke and J. Eggert. A two-stage correlation method for stereoscopic depth estimation. 
In International Conference on Digital Image Computing: Techniques and Applications (DICTA), pages 227–234, 2010.

To run the program, you need the following in root directory where source code files exists:
cudaSNCC.exe release version

source code files:
CudaImageFunctions.cu
ImageFunctions.cpp
ImageFunctions.h
produceDisparityMaps.cpp (main)

input directories containing initial stereo pairs: 
./training_image_0 
./training_image_1
left images of stereo pair in folder ./training_image_0/
right images of stereo pair in folder ./training_image_1/

output directories (need these in root directory): 
./disparity_GPU_SNCC5x5_13x13
./disparity_CPU_SNCC5x5_13x13 
./costmap_GPU_SNCC5x5_13x13
./costmap_CPU_SNCC5x5_13x13

miscellaneous other notes:
-For CPU, NCC window size and summation filter window size can be modified in main (default = 5x5 for NCC, 13x13 for summation).
-Output directory names should be changed to reflect updated window sizes.
-For GPU, NCC window size is fixed at 5 and summation filter window size is fixed at 13. Cannot be changed.
-The program takes no arguments.
-You can modify whether or not SNCC uses CPU or GPU on line 327 in main function of produceDisparityMaps.cpp
-Default / current operation mode = GPU.
-You can modify which KITTI sequences and frames are used to compute disparity maps in lines 345 - 348 of produceDisparityMaps.cpp
-The exclusive scan functions used in the GPU implementation produce one less disparity column and one less disparity row on the right / bottom of the output.
-The GPU version makes guesses for disparity where the CPU does not, so the results will not be 100% identical when CPU produces 0-valued disparities.
-KITTI training sequences 000000-000005 are included in this zip file for reference. This is meant as sample data only and is not the entire dataset.
