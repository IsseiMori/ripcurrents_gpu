#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "opencv2/cudaarithm.hpp"

#include "method.hpp"

using namespace std;

string getFileName(const string& s) {

   char sep = '/';

#ifdef _WIN32
   sep = '\\';
#endif

   size_t i = s.rfind(sep, s.length());
   if (i != string::npos) {
      return(s.substr(i+1, s.length() - i));
   }

   return(s);
}


method::method (string _file_name, 
				string _outfile_dir,
				int _height) {

	file_name = getFileName(_file_name);

	outfile_dir = _outfile_dir;
	
	video = cv::VideoCapture(_file_name);
	if (!video.isOpened())
	{
		cout << file_name << " File not found" << endl;
		exit(1);
	}

	fps = video.get(CAP_PROP_FPS);
	total_frame = video.get(CAP_PROP_FRAME_COUNT);

	height = _height;
	width = floor(video.get(cv::CAP_PROP_FRAME_WIDTH) * 
			height / video.get(cv::CAP_PROP_FRAME_HEIGHT));

	const int numLevels = 4;
	const float pyrScale = 0.5;
	const bool fastPyramids = true;
	const int winSize = 11;
	const int numIters = 10;
	const int polyN = 7; // 5 or 7
	const float polySigma = 2.4;
	dflow = cv::cuda::FarnebackOpticalFlow::create(numLevels, pyrScale, fastPyramids, winSize, numIters, polyN, polySigma);
			
}

VideoWriter* method::ini_video_output (string video_name) {
	
	VideoWriter* video_output = new VideoWriter (video_name + ".mp4", 
				  0x7634706d, 
				  fps, cv::Size(width,height),true);
	
	if (!video_output->isOpened())
	{
		cout << "!!! Output video could not be opened" << endl;
		return NULL;
	}

	return video_output;

}


void method::ini_buffer (int buffer_size) {
	current_buffer = 0;
	average_flow = Mat::zeros(height,width,CV_32FC2);

	for ( int i = 0; i < buffer_size; i++ )
	{
		buffer.push_back(Mat::zeros(height,width,CV_32FC2));
	}
}

void method::update_buffer (int buffer_size) {
	average_flow -= buffer[current_buffer] / static_cast<float>(buffer_size);
	buffer[current_buffer] = flow.clone();
	average_flow += buffer[current_buffer] / static_cast<float>(buffer_size);

	current_buffer++;
	if ( current_buffer >= buffer_size ) current_buffer = 0;
}

void convert_2dflow_to_3dmat(Mat& flow_x, Mat& flow_y) {

}

void method::calc_FB () {
	// recommended 5 5 1.1
	// recommended 5 7 1.5
	// no banding 20 (3) 15 1.2
	// calcOpticalFlowFarneback(prev_frame, curr_frame, flow, 0.5, 2, 20, 3, 15, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN);
	// calcOpticalFlowFarneback(prev_frame, curr_frame, flow, 0.5, 2, 5, 3, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);

	prev_frame_d.upload(prev_frame);
	curr_frame_d.upload(curr_frame);

	dflow->calc(prev_frame_d, curr_frame_d, flow_d);

	flow_d.download(flow);

}

void method::eliminate_std(int sig) {

	float mean = 0;
	float std = 0;
	int n = 0;
	float diff_sum = 0;

	flow.forEach<Pixel2>([&](Pixel2& px, const int pos[]) -> void {
		mean += sqrt(px.x * px.x + px.y*px.y);
		n++;
	});

	mean = mean / n;

	flow.forEach<Pixel2>([&](Pixel2& px, const int pos[]) -> void {
		diff_sum += pow(mean - sqrt(px.x * px.x + px.y*px.y),2);
	});

	std = sqrt(diff_sum / (n-1));

	flow.forEach<Pixel2>([&](Pixel2& px, const int pos[]) -> void {

		if ( sqrt(px.x * px.x + px.y*px.y) - mean > std * sig) {
			px.x = 0;
			px.y = 0;
		}

	});
}

void method::normalize_flow() {
	flow.forEach<Pixel2>([&](Pixel2& px, const int pos[]) -> void {

		float theta = atan2 (px.y, px.x);
		
		px.x = cos(theta);
		px.y = sin(theta);

	});
}

void method::vector_to_color(Mat& curr, Mat& out_img) {

	static float max_displacement = 0;
	float max_displacement_new = 0;

	float global_theta = 0;
	float global_magnitude = 0;

	for ( int row = 0; row < curr.rows; row++ ) {
		Pixel2* ptr = curr.ptr<Pixel2>(row, 0);
		Pixelc* ptr2 = out_img.ptr<Pixelc>(row, 0);

		for ( int col = 0; col < curr.cols; col++ ) {
			float theta = atan2(ptr->y, ptr->x)*180/M_PI;	// find angle
			theta += theta < 0 ? 360 : 0;	// enforce strict positive angle
			
			// store vector data
			ptr2->x = theta / 2;
			ptr2->y = 255;
			//ptr2->z = sqrt(ptr->x * ptr->x + ptr->y * ptr->y)*128/max_displacement+128;
            ptr2->z = sqrt(ptr->x * ptr->x + ptr->y * ptr->y)*255/max_displacement;
			//if ( ptr2->z < 30 ) ptr2->z = 0;

			// store the previous max to maxmin next frame
			if ( sqrt(ptr->x * ptr->x + ptr->y * ptr->y) > max_displacement_new ) max_displacement_new = sqrt(ptr->x * ptr->x + ptr->y * ptr->y);

			global_theta += ptr2->x * ptr2->z;
			global_magnitude += ptr2->z;


			ptr++;
			ptr2++;
		}
	}

	max_displacement = max_displacement_new;

	// show as hsv format
	cvtColor(out_img, out_img, COLOR_HSV2BGR);
}

void method::vector_to_dir_color(Mat& curr, Mat& out_img) {

	static float max_displacement = 0;
	float max_displacement_new = 0;

	float global_theta = 0;
	float global_magnitude = 0;

	for ( int row = 0; row < curr.rows; row++ ) {
		Pixel2* ptr = curr.ptr<Pixel2>(row, 0);
		Pixelc* ptr2 = out_img.ptr<Pixelc>(row, 0);

		for ( int col = 0; col < curr.cols; col++ ) {
			float theta = atan2(ptr->y, ptr->x)*180/M_PI;	// find angle
			theta += theta < 0 ? 360 : 0;	// enforce strict positive angle
			
			// store vector data
			ptr2->x = theta / 2;
			ptr2->y = 255;
			ptr2->z = 255;
			//if ( ptr2->z < 30 ) ptr2->z = 0;

			// store the previous max to maxmin next frame
			if ( sqrt(ptr->x * ptr->x + ptr->y * ptr->y) > max_displacement_new ) max_displacement_new = sqrt(ptr->x * ptr->x + ptr->y * ptr->y);

			global_theta += ptr2->x * ptr2->z;
			global_magnitude += ptr2->z;


			ptr++;
			ptr2++;
		}
	}

	max_displacement = max_displacement_new;

	// show as hsv format
	cvtColor(out_img, out_img, COLOR_HSV2BGR);
}

// Load the previous frame
// Named ini_frame because calling read_frame twice is confusing
int method::ini_frame () {
	return read_frame();
}

int method::read_frame () {

	Mat frame, grayscaled_frame;

	curr_frame.copyTo (prev_frame);
	video.read (frame);
	if (frame.empty()) return 1;
	resize (frame, resized_frame, Size(width, height), 0, 0, INTER_LINEAR);
	cvtColor (resized_frame, grayscaled_frame, COLOR_BGR2GRAY);
	grayscaled_frame.copyTo(curr_frame);
	return 0;
}

void method::drawFrameCount (Mat& outImg, int framecount) {
	putText(outImg, to_string(framecount), Point(30,30), 
	FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(250,250,250), 1, false);
}

void method::ini_draw_colorwheel () {
	colorwheel = imread("colorWheel.jpg");
    resize(colorwheel, colorwheel, Size(height/8, height/8));
}

void method::draw_colorwheel(Mat& out_img) {
	Mat mat = (Mat_<double>(2,3)<<1.0, 0.0, width - height/8, 0.0, 1.0, 0);
    warpAffine(colorwheel, out_img, mat, out_img.size(), INTER_LINEAR, cv::BORDER_TRANSPARENT);
}