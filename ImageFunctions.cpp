#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
//#include "ImageFunctions.h
using namespace std;


////////////////////////
// changed read_pgm_image, write_pgm_Uimage functions to work with 2005 version of ifstream
//


unsigned char *read_pgm_image(char *fname, int *nx, int *ny)
{
	unsigned char *dummy;
	int i_size, x, y, s;
	ifstream fin;

	fin.open(fname, ios::binary);
	if (fin.fail()){
		cout << "File I/O error" << endl;
		exit(0);
	}


	char line[110];
	fin.getline(line,100); //read P#
	//printf(line);

	fin.getline(line, 5, ' ');
	//printf(line);

	if(line[0] >= '0' && line[0] <= '9')  // read number
	{
		//printf("\n no comment\n");
		//printf(line);

		x = atoi(line);
	}
	else
	{
		//printf("\n  something else \n");
		//printf(line);
		fin.getline(line, 100);  // read through comment
		fin.getline(line, 5, ' ');
		//printf(line);
		x = atoi(line);
	}

	//fin >> y >> s; // >> t;
	fin.getline(line, 50);
	y = atoi(line);
	fin.getline(line, 50);
	s = atoi(line);
	//	cout << "x " << x << " y " << y << endl;
	//cout << "s " << s << endl; // " t " << t << endl;
	while('\n'==fin.peek())
		fin.ignore(1);
	*nx=x; *ny=y;
	i_size=(*nx)*(*ny);
	dummy = new unsigned char [i_size];
	fin.read((char *)dummy, i_size);
	fin.close();

	return(dummy);
}

int write_pgm_Uimage(unsigned char *image, char *fname, int nx, int ny)
{
	ofstream fout;

	//cout << "Writing " << nx << "*" << ny << " image." << endl;

	fout.open(fname, ios::binary);
	if (fout.fail()) {
		cout << "File I/O error" << endl;
		return(0);
	}

	fout << "P5" << endl;
	fout << "# Intermediate image file" << endl;
	fout << nx << " ";
	fout << ny << endl;
	fout << "255" << endl ;


	fout.write((const char *)image, nx * ny * sizeof(unsigned char));
	fout.close();

	return 1;
}

void arrayFloatToBinaryFile(float *m, int size, std::ofstream& oF){
	if(!oF){
		std::cout << "File I/O Error";
		throw 1;
	}

	float cflt;
	//std::cout << "saved to bin:" << endl;
	for(int j = 0; j < size; j++){
			cflt = m[j];
			//std::cout << cflt << "\t ";
			oF.write( reinterpret_cast<char*>( &cflt ), sizeof cflt );
	}
}

void binaryFileToArrayFloat(std::ifstream& iF, float* array, int size){
	float* m = new float[size];
	float read;
	int j = 0;

	if(!iF) {
		std::cout << "File I/O Error";
		throw 1;
	}
	//std::cout << "read from bin:" << endl;

	while(j < size && !iF.eof()) {
			iF.read( reinterpret_cast<char*>( &read ), sizeof read );
			m[j] = read;
			//std::cout << read << "\t ";
			j++;
	}

	if(j < size) {
		std::cout << "premature end of file while reading..." << std::endl;
		throw 1;
	}

	for (j = 0; j < size; j++ ){
			array[j] = m[j];
	}
	//clean-up array
	delete[] m;
}

//accounts for the way that matlab writes PGM images, which is different from Irfanview (first three lines differ)
unsigned char *read_pgm_image_matlab(char *fname, int *nx, int *ny)
{
	unsigned char *dummy;
	int i_size, x, y;
	ifstream fin;

	fin.open(fname, ios::binary);
	if (fin.fail()){
		cout << "File I/O error" << endl;
		exit(0);
	}

	char line[110];
	fin.getline(line,100); //get line image details

	char xdim[6];
	for (int p = 0; p < 4; p++){
		xdim[p] = line[3+p];
	}
	xdim[5] = '\0';
	x = atoi(xdim);

	char ydim[5];
	for (int p = 0; p < 3; p++){
		ydim[p] = line[8+p];
	}
	ydim[4] = '\0';
	y = atoi(ydim);

	while('\n'==fin.peek())
		fin.ignore(1);
	*nx=x; *ny=y;
	i_size=(*nx)*(*ny);
	dummy = new unsigned char [i_size];
	fin.read((char *)dummy, i_size);
	fin.close();

	return(dummy);
}
