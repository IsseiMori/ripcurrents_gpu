#include <string>
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream> 
#include <time.h>

#include <opencv2/opencv.hpp>

#include "fn_convert.hpp"

using namespace std;

fn_convert::fn_convert (string _file_name, 
							string _outfile_dir,
							int _height)
							: method(_file_name, _outfile_dir, _height) {
}



void fn_convert::run () {
	cout << "Running color map" << endl;

	VideoWriter* video_output_color = ini_video_output (outfile_dir);

	cout << fps << endl;

	ini_frame();

	for (int framecount = 1; true; ++framecount) {

		if (read_frame()) break;

		cout << "frame: " << framecount << endl;

		Mat out_img;
		resized_frame.copyTo(out_img);

		
		
		imshow ("grid_buoy color map", out_img);
		video_output_color->write (out_img);

		curr_frame.copyTo (prev_frame);
		if ( waitKey(1) == 27) break;

	}

	// clean up
	video_output_color->release();
	destroyAllWindows();

}
