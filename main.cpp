#include <thread>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>

#include <opencv2/opencv.hpp>

#include "tinyply.h"

using namespace tinyply;
using namespace std;
using namespace cv;

void read_ply_file(const std::string & filename, vector<float> &verts, vector<uint32_t> &faces)
{
	// Tinyply can and will throw exceptions at you!
	try
	{
		// Read the file and create a std::istringstream suitable
		// for the lib -- tinyply does not perform any file i/o.
		std::ifstream ss(filename);

		// Parse the ASCII header fields
		PlyFile file(ss);

		for (auto e : file.get_elements())
		{
			std::cout << "element - " << e.name << " (" << e.size << ")" << std::endl;
			for (auto p : e.properties)
			{
				std::cout << "\tproperty - " << p.name << " (" << PropertyTable[p.propertyType].str << ")" << std::endl;
			}
		}
		std::cout << std::endl;

		for (auto c : file.comments)
		{
			std::cout << "Comment: " << c << std::endl;
		}

		uint32_t vertexCount, faceCount;
		vertexCount = faceCount = 0;

		vertexCount = file.request_properties_from_element("vertex", { "x", "y", "z" }, verts);
		faceCount = file.request_properties_from_element("face", { "vertex_indices" }, faces, 3);

		file.read(ss);

		std::cout << "\tRead " << verts.size() << " total vertices (" << vertexCount << " properties)." << std::endl;
		std::cout << "\tRead " << faces.size() << " total faces (triangles) (" << faceCount << " properties)." << std::endl;
		
	}

	catch (const std::exception & e)
	{
		std::cerr << "Caught exception: " << e.what() << std::endl;
	}
}

const int laser_slider_max = 200;
int laser_slider;
double laser;

float floor_height = 1.6;
const float resolution = 0.01; // meters per pixel
float offsetx = 0, offsetz = 0;

int height,width;
uint32_t vertexCount, faceCount;

vector<float> verts;
vector<uint32_t> faces;

Mat fake;

int occ[1000][1000];

void on_trackbar( int , void *)
{
	laser = (float) laser_slider / 100;

	cout << "Start_cal " << laser << endl;

	Mat grid(height, width, CV_8UC1, 128 );
	memset(occ, 0, sizeof(occ));
	cout << offsetx << endl;
	for ( int i = 0; i < vertexCount; ++i)
	{
		if ( verts[3*i + 1] - floor_height > 0.1)	continue;

		float x = verts[3*i] + offsetx;
		float z = verts[3*i + 2] + offsetz;

		int r = (int) (x / resolution);
		int c = (int) (z / resolution);

		if ( r >= 898 || c >= 1055 )
		{
			cout << r << " " << c << endl;
		}
		if (abs(floor_height - verts[3*i + 1] - laser) < 0.1 )
		{
			occ[r][c] = 2;
			//cout << x << " " << z << " " << r << " " << c << endl;
		}
		else if (occ[r][c] == 0 and abs(floor_height - verts[3*i + 1]) < 0.05 )
		{
			occ[r][c] = 1;
		}
	}

	cout << "Draw Image" << endl;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if ( occ[i][j] == 2 )	{
				grid.at<uchar>(i,j) = 0;
			}
		}
	}

	cout << "End Draw" << endl;
	imshow("Occupancy grid", grid);

}

int main (int argc, char** argv)
{

	if (argc < 2 )	return 0;
	
	string fileName = argv[1];	
    
    read_ply_file(fileName, verts, faces);

    std::cout << "\tRead " << verts.size() << " total vertices (" << std::endl;
	std::cout << "\tRead " << faces.size() << " total faces (triangles) (" << std::endl;

	
	vertexCount = verts.size() / 3;

	float ground = 10, top = 0;

	float sizex = 0, sizez = 0;

	for ( int i = 0; i < vertexCount; ++i)
	{
		ground = min( ground, verts[3*i+1]);
		top = max(top, verts[3*i+1]);

		offsetx = min(offsetx, verts[3*i]);
		sizex   = max(sizex, verts[3*i]);
		offsetz = min(offsetz, verts[3*i + 2]);
		sizez   = max(sizez, verts[3*i + 2]);
	}

	std::cout << "The minimum point is " << ground << std::endl;
	std::cout << "The highest point is " << top << std::endl;

	offsetx *= -1;
	offsetz *= -1;
	
	sizex += offsetx;
	sizez += offsetz;

	cout << "Height " << sizex << " Width " << sizez << endl;

	height = (int) (sizex / resolution + 10);
	width = (int) (sizez / resolution + 10);
	cout << "Height " << height << " Width " << width << endl;

	

	laser_slider = 30;
	
	 /// Create Windows
 	namedWindow("Occupancy grid", WINDOW_AUTOSIZE);

 	/// Create Trackbars
 	char TrackbarName[50];
 	sprintf( TrackbarName, "Laser height max %d cm", laser_slider_max );

 	createTrackbar( TrackbarName, "Occupancy grid", &laser_slider, laser_slider_max, on_trackbar );

	cout << " Track bar Complete" << endl;
 	/// Show some stuff
 	on_trackbar( laser_slider, 0 );
	while(true)
	{
		char c = waitKey(0);
		if ( c == 'q')	break;
	}
}