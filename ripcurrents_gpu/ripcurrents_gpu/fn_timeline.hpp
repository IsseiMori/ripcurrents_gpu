#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "method.hpp"

using namespace std;

typedef cv::Point_<float> Pixel2;

class timeline {
	private:
		vector<Pixel2> vertices;
	public: 
		timeline(){};
		timeline(Pixel2 start, Pixel2 end, int vertices_count, int _die_at);

		// run LK method on each vertex and draw lines
		void runLK(Mat u_prev, Mat u_curr, Mat& out_img, bool isNorm);
		int die_at;

		// void runFarneBack();
};

class fn_timeline: public method {
	private:
		vector<timeline> timelines;
		int vnum;
		int born_period;
		int lifespan;

	public:
		fn_timeline (string file_name,
					 string _outfile_dir,
					 int _height, 
					 int _vnum = 10,
					 int _born_period = 0,
					 int _lifespan = 0);
		void run(bool isNorm);
		void add_timeline (Pixel2 start, Pixel2 end, int vertices_count, int die_at);
		int get_vnum () {return vnum;}
		vector<pair<Pixel2,Pixel2>> start_end;
};