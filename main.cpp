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

#include <Eigen/Dense>

#include "tinyply.h"

using namespace tinyply;
using namespace std;
using namespace cv;
using namespace Eigen;
using namespace pcl;

const int laser_slider_max = 200;
int laser_slider;
double laser;

const float resolution = 0.01; // meters per pixel
float offsetx = 0, offsetz = 0;

int height,width;
uint32_t vertexCount, faceCount;

const bool enTransform = true;

vector<PointXYZ> verts;
vector<uint32_t> faces;

Mat fake;

int occ[1000][1000];

void read_ply_file(const std::string & filename)
{
	// Tinyply can and will throw exceptions at you!
	try
	{
		// Read the file and create a std::istringstream suitable
		// for the lib -- tinyply does not perform any file i/o.
		std::ifstream ss(filename);
		vector<float> points;

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

		vertexCount = file.request_properties_from_element("vertex", { "x", "y", "z" }, points);
		faceCount = file.request_properties_from_element("face", { "vertex_indices" }, faces, 3);

		file.read(ss);

		for ( int i = 0; i < points.size(); i+=3 )
		{
			PointXYZ p;
			p.x = points[i];
			p.y = points[i+1];
			p.z = points[i+2];
			verts.push_back(p);
		}

		std::cout << "\tRead " << verts.size() << " total vertices (" << vertexCount << " properties)." << std::endl;
		std::cout << "\tRead " << faces.size() << " total faces (triangles) (" << faceCount << " properties)." << std::endl;
		
	}

	catch (const std::exception & e)
	{
		std::cerr << "Caught exception: " << e.what() << std::endl;
	}
}

void on_trackbar( int , void *)
{
	laser = (float) laser_slider / 100;

	cout << "Start_cal " << laser << endl;

	Mat grid(height, width, CV_8UC1, 128 );
	memset(occ, 0, sizeof(occ));
	cout << offsetx << endl;
	for ( int i = 0; i < vertexCount; ++i)
	{
		if ( verts[i].y < 0)	continue;

		float x = verts[i].x + offsetx;
		float z = verts[i].z + offsetz;

		int r = (int) (x / resolution);
		int c = (int) (z / resolution);

		if ( r >= 898 || c >= 1055 )
		{
			cout << r << " " << c << endl;
		}
		if (abs(verts[i].y - laser) < 0.1 )
		{
			occ[r][c] = 2;
			//cout << x << " " << z << " " << r << " " << c << endl;
		}
		else if (occ[r][c] == 0 and verts[i].y < 0.05 )
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
//0.13184 1.5674 -0.94338
//0.99496 2.083 1.8732
void cut_plane(vector<PointXYZ> &all_point, vector<PointXYZ> &remain)
{
	remain.clear();
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	cloud->width  = all_point.size();
	cloud->height = 1;
	cloud->points.resize (cloud->width * cloud->height);

	// Generate the data
	for (size_t i = 0; i < cloud->points.size (); ++i)
	{
		cloud->points[i] = all_point[i];
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

	// Print Origin and another point

	cout << all_point[inliers->indices[0]].x << " "
		<< all_point[inliers->indices[0]].y << " "
		<< all_point[inliers->indices[0]].z << endl;

	cout << all_point[inliers->indices[inliers->indices.size()-1]].x << " "
		<< all_point[inliers->indices[inliers->indices.size()-1]].y << " "
		<< all_point[inliers->indices[inliers->indices.size()-1]].z << endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);

	new_cloud->width  = all_point.size() - inliers->indices.size();
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
			remain.push_back(cloud->points[i]);
		}
	}

	pcl::io::savePCDFileASCII ("deleted_plane.pcd", *new_cloud);
	pcl::io::savePCDFileASCII ("plane.pcd", *plane);
}

void findRotationMatrix(vector<float> &equation, PointXYZ origin, PointXYZ pointInPlane, Matrix4f &mat)
{
	Matrix4f wTOF;
	
	// Find X axis
	Vector3f x(pointInPlane.x - origin.x, pointInPlane.y - origin.y, pointInPlane.z - origin.z);
	x = x / x.norm();

	// Find Y axis
	Vector3f y(equation[0], equation[1], equation[2]);
	y = y / y.norm();

	Vector3f z;
	z = x.cross(y);
	z = z / z.norm();

	cout << x << endl;
	cout << y << endl;
	cout << z << endl;

	wTOF.block(0,0,3,1) = x;
	wTOF.block(0,1,3,1) = y;
	wTOF.block(0,2,3,1) = z;
	wTOF(0,3) = origin.x;
	wTOF(1,3) = origin.y;
	wTOF(2,3) = origin.z;
	wTOF(3,3) = 1;
	wTOF(3,0) = wTOF(3,1) = wTOF(3,2) = 0;

	cout << wTOF << endl;

	mat = wTOF.inverse();

	cout << mat << endl;
}

int main (int argc, char** argv)
{

	if (argc < 2 )	return 0;
	
	string fileName = argv[1];	
  
    read_ply_file(fileName);

    std::cout << "\tRead " << verts.size() << " total vertices (" << std::endl;
	std::cout << "\tRead " << faces.size() << " total faces (triangles) (" << std::endl;

	if (enTransform)
	{
		// plane -0.00455174 -0.988815 0.149075 1.78232
		//0.13184 1.5674 -0.94338
		//0.99496 2.083 1.8732
		PointXYZ origin(0.13184, 1.5674, -0.94338),p(0.99496, 2.083, 1.8732);
		vector<float> equation(5);
		equation[0] = -0.00455174;
		equation[1] = -0.988815;
		equation[2] = 0.149075;
		equation[3] = 1.78232;

		Matrix4f R;
		findRotationMatrix(equation, origin, p, R);

		for (int i=0; i < verts.size(); i++ )
		{
			Vector4f p(verts[i].x, verts[i].y, verts[i].z, 1);
			p = R * p;
			verts[i].x = p[0];
			verts[i].y = p[1];
			verts[i].z = p[2];
		}

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

		cloud->width  = verts.size();
		cloud->height = 1;
		cloud->points.resize (cloud->width * cloud->height);

		// Generate the data
		for (size_t i = 0; i < cloud->points.size (); ++i)
		{
			cloud->points[i] = verts[i];
		}

		pcl::io::savePCDFileASCII ("roteted_mao.pcd", *cloud);
	}
	
	vertexCount = verts.size();


	float sizex = 0, sizez = 0;

	for ( int i = 0; i < vertexCount; ++i)
	{

		offsetx = min(offsetx, verts[i].x);
		sizex   = max(sizex, verts[i].x);
		offsetz = min(offsetz, verts[i].z);
		sizez   = max(sizez, verts[i].z);
	}

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
	vector<PointXYZ> all,remain;
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