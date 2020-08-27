#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "method.hpp"

using namespace std;


class fn_convert: public method {
	private:
	public:
		fn_convert (string _file_name,
					  string _outfile_dir,
					  int _height);
		void run ();
};