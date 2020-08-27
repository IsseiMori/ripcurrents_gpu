// ripcurrents_gpu.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>

#include <opencv2/opencv.hpp>

#include <opencv2/core/utility.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include "opencv2/video.hpp"
#include "opencv2/cudaarithm.hpp"

#include "method.hpp"
#include "fn_dir_color.hpp"
#include "fn_timeline.hpp"
#include "fn_convert.hpp"

const int numLevels = 4;
const float pyrScale = 0.5;
const bool fastPyramids = true;
const int winSize = 11;
const int numIters = 10;
const int polyN = 7; // 5 or 7
const float polySigma = 2.4;
const int flags = 0;
const bool resize_img = false;
const float rfactor = 2.0;

using namespace std;
using namespace cv;


template <typename T>
inline T mapVal(T x, T a, T b, T c, T d)
{
	x = ::max(::min(x, b), a);
	return c + (d - c) * (x - a) / (b - a);
}

static void colorizeFlow(const Mat& u, const Mat& v, Mat& dst)
{
	double uMin, uMax;
	minMaxLoc(u, &uMin, &uMax, 0, 0);
	double vMin, vMax;
	minMaxLoc(v, &vMin, &vMax, 0, 0);
	uMin = ::abs(uMin); uMax = ::abs(uMax);
	vMin = ::abs(vMin); vMax = ::abs(vMax);
	float dMax = static_cast<float>(::max(::max(uMin, uMax), ::max(vMin, vMax)));
	dst.create(u.size(), CV_8UC3);
	for (int y = 0; y < u.rows; ++y)
	{
		for (int x = 0; x < u.cols; ++x)
		{
			dst.at<uchar>(y, 3 * x) = 0;
			dst.at<uchar>(y, 3 * x + 1) = (uchar)mapVal(-v.at<float>(y, x), -dMax, dMax, 0.f, 255.f);
			dst.at<uchar>(y, 3 * x + 2) = (uchar)mapVal(u.at<float>(y, x), -dMax, dMax, 0.f, 255.f);
		}
	}
}

int main(int argc, char **argv)
{
	string file_name = argv[1];
	cout << file_name << endl;

	string outfile_dir = argv[2];

	int option = 0;
	if (argc >= 4)
	{
		option = stoi(argv[3]);
	}

	clock_t start = clock();

	switch (option) {

		case -1: {
			if (argc < 5) {
				fn_convert convert = fn_convert(file_name, outfile_dir, 480);
				convert.run();
			}
			else {
				fn_convert convert = fn_convert(file_name, outfile_dir, 0);
				convert.run();
			}
			break;
		}

		// timeline
		case 0: {
			cout << "Click two end points of the timeline, then press any key to start" << endl;
			if (argc < 7) {
				fn_timeline timeline = fn_timeline(file_name, outfile_dir, 480, 20, 0, 0);
				timeline.run(false);
			}
			else if (argc == 7) {
				fn_timeline timeline = fn_timeline(file_name, outfile_dir, 480, stoi(argv[4]), stoi(argv[5]), stoi(argv[6]));
				timeline.run(false);
			}
			else if (argc > 7) {
				fn_timeline timeline = fn_timeline(file_name, outfile_dir, 480, stoi(argv[4]), stoi(argv[5]), stoi(argv[6]));
				timeline.run(true);
			}
			break;
		}

		// color field
		case 5: {
			fn_dir_color dir_color = fn_dir_color(file_name, outfile_dir, 480);

			if (argc < 5) {
				dir_color.run();
			}
			else {
				dir_color.run(stoi(argv[4]));
			}
			break;
		}

		// direction only color
		case 6: {
			fn_dir_color dir_color = fn_dir_color(file_name, outfile_dir, 480);
			if (argc < 5) {
				dir_color.run_dir();
			}
			else {
				dir_color.run_dir(stoi(argv[4]));
			}
			break;
		}

		// normalized flow color
		case 7: {
			fn_dir_color dir_color = fn_dir_color(file_name, outfile_dir, 480);

			if (argc < 5) {
				dir_color.run_norm();
			}
			else {
				dir_color.run_norm(stoi(argv[4]));
			}
			break;
		}

		// normalized flow color
		case 15: {
			fn_dir_color dir_color = fn_dir_color(file_name, outfile_dir, 480);

			if (argc < 5) {
				dir_color.run_norm_rgb();
			}
			else {
				dir_color.run_norm_rgb(stoi(argv[4]));
			}
			break;
		}

		// normalized flow color
		case 16: {
			fn_dir_color dir_color = fn_dir_color(file_name, outfile_dir, 480);

			if (argc < 5) {
				dir_color.run_rgb();
			}
			else {
				dir_color.run_rgb(stoi(argv[4]));
			}
			break;
		}

		// normalized flow color with mask
		case 17: {
			cout << "normalized flow color with mask" << endl;
			fn_dir_color dir_color = fn_dir_color(file_name, outfile_dir, 480);
			cout << argv[4] <<  stoi(argv[5]) << endl;
			dir_color.run_norm_mask(argv[4], stoi(argv[5]));
			break;
		}
	}



	clock_t end = clock();
	const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
	printf("time %lf[ms]\n", time);

	return 0;

}