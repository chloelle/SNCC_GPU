#ifndef ImageFunctions_H
#define ImageFunctions_H

unsigned char *read_pgm_image(char *fname, int *nx, int *ny);
int write_pgm_image(int *image, char *fname, int nx, int ny);
int write_pgm_Uimage(unsigned char *image, char *fname, int nx, int ny);

void arrayFloatToBinaryFile(float *m, int size, std::ofstream& oF); //CL 4-22-2014
void binaryFileToArrayFloat(std::ifstream& iF, float* array, int size); //CL 4-22-2014

unsigned char *read_pgm_image_matlab(char *fname, int *nx, int *ny);

#endif

