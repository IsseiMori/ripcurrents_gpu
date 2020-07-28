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
	}



	clock_t end = clock();
	const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
	printf("time %lf[ms]\n", time);

	return 0;

	//std::cout << getBuildInformation() << std::endl;
	//cuda::printShortCudaDeviceInfo(cuda::getDevice());

	//cuda::GpuMat frame0GPU, frame1GPU, flowGPU;
	//Mat frame0_rgb, frame1_rgb, frame0, frame1;
	//Mat flowx, flowy, flow;
	//int nframes = 0, width = 0, height = 0;

	//// Create OpenCV windows
	//namedWindow("Dense Flow", WINDOW_AUTOSIZE);

	//// Create the optical flow object
	//Ptr<cuda::FarnebackOpticalFlow> dflow = cuda::FarnebackOpticalFlow::create(numLevels, pyrScale, fastPyramids, winSize, numIters, polyN, polySigma);

	//VideoCapture cap("E:/ripcurrents/rip_currents_with_breaking_waves/rip_01_fast.mp4");
	//if (cap.isOpened() == 0) {
	//	return -1;
	//}

	//cap >> frame1_rgb;

	//width = frame1_rgb.cols;
	//height = frame1_rgb.rows;

	//int w, h;
	//w = 480;
	//h = (float)height * (float)w / (float)width;

	//resize(frame1_rgb, frame1_rgb, Size(w, h), 0, 0, INTER_LINEAR);
	//cvtColor(frame1_rgb, frame1, COLOR_BGR2GRAY);

	//while (frame1.empty() == false)
	//{
	//	std::cout << nframes << std::endl;

	//	if (nframes >= 1)
	//	{
	//		frame0GPU.upload(frame0);
	//		frame1GPU.upload(frame1);

	//		dflow->calc(frame0GPU, frame1GPU, flowGPU);

	//		cuda::GpuMat planes[2];
	//		cuda::split(flowGPU, planes);
	//		planes[0].download(flowx);
	//		planes[1].download(flowy);

	//		colorizeFlow(flowx, flowy, flow);

	//		imshow("Dense Flow", flow);

	//		waitKey(3);

	//	}

	//	frame1_rgb.copyTo(frame0_rgb);
	//	resize(frame0_rgb, frame0_rgb, Size(w, h), 0, 0, INTER_LINEAR);
	//	cvtColor(frame0_rgb, frame0, COLOR_BGR2GRAY);

	//	nframes++;

	//	cap >> frame1_rgb;
	//	resize(frame1_rgb, frame1_rgb, Size(w, h), 0, 0, INTER_LINEAR);

	//	if (frame1_rgb.empty() == false)
	//	{
	//		cvtColor(frame1_rgb, frame1, COLOR_BGR2GRAY);
	//	}
	//	else
	//	{
	//		break;
	//	}
	//}

	//destroyAllWindows();
}