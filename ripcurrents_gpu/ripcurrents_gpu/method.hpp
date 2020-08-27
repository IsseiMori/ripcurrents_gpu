#pragma once

#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>

using namespace cv;
using namespace std;

#define M_PI 3.14159265358979

typedef cv::Point3_<uchar> Pixelc;
typedef cv::Point_<float> Pixel2;

class method {
	protected:
		// file info
		VideoCapture video;
		string file_name;
		string outfile_dir;
		double fps;
		int total_frame;
		bool do_resize;

		// frame rgb Mat
		Mat curr_frame, prev_frame;
		cuda::GpuMat curr_frame_d, prev_frame_d;
		Mat resized_frame;

		Mat colorwheel;


	// Optical Flow
	protected:
		Mat flow;
		cuda::GpuMat flow_d;

		int current_buffer;
		vector<Mat> buffer;
		Mat average_flow;

		Ptr<cv::cuda::FarnebackOpticalFlow> dflow;

	public:
		int width;
		int height;

		method (string file_name, string _outfile_dir, int _height);
		
		// Video
		VideoWriter* ini_video_output (string video_name);

		// Buffer
		void ini_buffer (int buffer_size);
		void update_buffer (int buffer_size);

		// read frames
		int ini_frame ();
		int read_frame ();

		// optical flow
		void calc_FB ();

		void eliminate_std(int sig);
		void normalize_flow();
		void flow_mask_ocean(Mat& mask);

		// flow to color
		void vector_to_color(Mat& curr, Mat& out_img);
		void vector_to_color2(Mat& curr, Mat& out_img); // xy to GB
		void vector_to_dir_color(Mat& curr, Mat& out_img);

		// draw helper
		void drawFrameCount (Mat& outImg, int framecount);
		void ini_draw_colorwheel ();
		void draw_colorwheel (Mat& out_img);

};