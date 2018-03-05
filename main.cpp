#include <thread>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>

#include <opencv2/opencv.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

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
// plane -0.00455174 -0.988815 0.149075 1.78232
void cut_plane(vector<float> &all_point, vector<float> &remain)
{
	remain.clear();
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	cloud->width  = all_point.size() / 3;
	cloud->height = 1;
	cloud->points.resize (cloud->width * cloud->height);

	// Generate the data
	for (size_t i = 0; i < cloud->points.size (); ++i)
	{
		cloud->points[i].x = all_point[3*i];
		cloud->points[i].y = all_point[3*i + 1];
		cloud->points[i].z = all_point[3*i + 2];
	}
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	// Optional
	seg.setOptimizeCoefficients (true);
	// Mandatory
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setDistanceThreshold (0.1);

	seg.setInputCloud (cloud);
	seg.segment (*inliers, *coefficients);

	std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3] << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);

	new_cloud->width  = all_point.size() / 3 - inliers->indices.size();
	new_cloud->height = 1;
	new_cloud->points.resize (new_cloud->width * new_cloud->height);

	plane->width  = inliers->indices.size();
	plane->height = 1;
	plane->points.resize (plane->width * plane->height);

	int ind = 0,p = 0;
	for (int i = 0; i < cloud->points.size(); i++)
	{
		if ( ind < inliers->indices.size() and inliers->indices[ind] == i)
		{
			plane->points[ind] = cloud->points[i];
			ind++;
		}
		else
		{
			new_cloud->points[p] = cloud->points[i];
			p++;
			remain.push_back(cloud->points[i].x);
			remain.push_back(cloud->points[i].y);
			remain.push_back(cloud->points[i].z);
		}
	}

	pcl::io::savePCDFileASCII ("deleted_plane.pcd", *new_cloud);
	pcl::io::savePCDFileASCII ("plane.pcd", *plane);
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
	vector<float> all,remain;
	all = verts;
	while(true)
	{
		char c = waitKey(0);
		if ( c == 'q')	break;
		if ( c == 'c')	
		{
			cut_plane(all, remain);
			all = remain;
		}
	}
}