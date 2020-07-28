#include <string>
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream> 
#include <time.h>

#include <opencv2/opencv.hpp>

#include "fn_dir_color.hpp"

using namespace std;

fn_dir_color::fn_dir_color (string _file_name, 
							string _outfile_dir,
							int _height)
							: method(_file_name, _outfile_dir, _height) {
}


void fn_dir_color::justrun () {
	cout << "Running color map" << endl;

	ini_frame();

	clock_t start = clock();

	for (int framecount = 1; true; ++framecount) {
		cout << "Frame " << framecount << endl;
		if (read_frame()) break;

		calc_FB ();

		if ( waitKey(1) == 27) break;

	}

	clock_t end = clock();
    const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("optflow time %lf[ms]\n", time);

	// clean up
	destroyAllWindows();

}

void fn_dir_color::run (int buffer_size) {
	cout << "Running color map" << endl;

	VideoWriter* video_output_color = ini_video_output (outfile_dir + "_dir_color_" + "_color");
	VideoWriter* video_output_overlay = ini_video_output (outfile_dir + "_dir_color_" + "_overlay");

	cout << outfile_dir + "_dir_color_" + to_string(buffer_size) + "_color" << endl;

	ini_frame();
	ini_buffer(buffer_size);
	ini_draw_colorwheel ();

	for (int framecount = 1; true; ++framecount) {

		if (read_frame()) break;

		cout << "frame: " << framecount << endl;

		calc_FB ();

		Mat out_img;
		Mat out_img_overlay;
		resized_frame.copyTo(out_img);
		resized_frame.copyTo(out_img_overlay);

		eliminate_std (5);

		update_buffer (buffer_size);

		vector_to_color (average_flow, out_img);

		drawFrameCount(out_img, framecount);

		addWeighted( out_img, 1.0, out_img_overlay, 1.0, 0.0, out_img_overlay);

		// draw_colorwheel (out_img);
		draw_colorwheel (out_img_overlay);
		
		
		imshow ("grid_buoy color map", out_img);
		imshow ("grid_buoy overlay", out_img_overlay);
		video_output_color->write (out_img);
		video_output_overlay->write (out_img_overlay);

		curr_frame.copyTo (prev_frame);
		if ( waitKey(1) == 27) break;

	}

	// clean up
	video_output_color->release();
	video_output_overlay->release();
	destroyAllWindows();

}

void fn_dir_color::run_norm (int buffer_size) {
	cout << "Running color map" << endl;

	VideoWriter* video_output_color = ini_video_output (file_name + "_norm_color_" + to_string(buffer_size) + "_color");
	VideoWriter* video_output_overlay = ini_video_output (file_name + "_norm_color_" + to_string(buffer_size) + "_overlay");

	ini_frame();
	ini_buffer(buffer_size);
	ini_draw_colorwheel ();

	for (int framecount = 1; true; ++framecount) {

		if (read_frame()) break;

		calc_FB ();

		Mat out_img;
		Mat out_img_overlay;
		resized_frame.copyTo(out_img);
		resized_frame.copyTo(out_img_overlay);

		eliminate_std(5);
		normalize_flow();

		update_buffer (buffer_size);

		vector_to_color (average_flow, out_img);
		drawFrameCount(out_img, framecount);


		addWeighted( out_img, 1.0, out_img_overlay, 1.0, 0.0, out_img_overlay);

		draw_colorwheel (out_img);
		draw_colorwheel (out_img_overlay);
		
		
		imshow ("grid_buoy color map", out_img);
		imshow ("grid_buoy overlay", out_img_overlay);
		video_output_color->write (out_img);
		video_output_overlay->write (out_img_overlay);

		curr_frame.copyTo (prev_frame);
		if ( waitKey(1) == 27) break;

	}

	// clean up
	video_output_color->release();
	video_output_overlay->release();
	destroyAllWindows();

}

void find_incoming_dir (Mat& curr, Mat& flow_norm, Mat& out_img) {


	for ( int row = 0; row < curr.rows; row++ ) {
		Pixel2* ptr = curr.ptr<Pixel2>(row, 0);
		for ( int col = 0; col < curr.cols; col++ ) {
			ptr++;
		}
	}

	int hist[6] = {0,0,0,0,0,0};

	float ave_magnitude = 0;

	// Create a histgram and find average magnitude
	for ( int row = 0; row < curr.rows; row++ ) {
		Pixel2* ptr = curr.ptr<Pixel2>(row, 0);
		for ( int col = 0; col < curr.cols; col++ ) {
			ave_magnitude += sqrt(ptr->x * ptr->x 
							+ ptr->y * ptr->y);
			ptr++;
		}
	}

	ave_magnitude /= static_cast<float>(curr.rows * curr.cols);

	// Create a histgram and find average magnitude
	for ( int row = 0; row < curr.rows; row++ ) {
		Pixel2* ptr = curr.ptr<Pixel2>(row, 0);
		for ( int col = 0; col < curr.cols; col++ ) {
			float magnitude = sqrt(ptr->x * ptr->x + ptr->y * ptr->y);

			if (magnitude > ave_magnitude * 0.5) {
				int bin = static_cast<int>((atan2 (ptr->x, ptr->y) * 180 / M_PI + 180) / 360 * 6);
				if (bin == 6) bin = 0;
				hist[bin]++;
			}
			ptr++;
		}
	}

	int max_hist = 0;
	int max_id = 0;
	//cout << "BIN" << endl;
	for (int i = 0; i < 6; ++i) {
		if (hist[i] > max_hist) {
			max_id = i;
			max_hist = hist[i];
		}
	}

	int oppose1 = (max_id + 3 > 5) ? max_id - 3 : max_id + 3;
	int opp_near1 = (max_id + 4 > 5) ? max_id - 2 : max_id + 4;
	int opp_near2 = (max_id + 2 > 5) ? max_id - 4 : max_id + 2;
	int max_near1 = (max_id + 1 > 5) ? max_id - 5 : max_id + 1;
	int max_near2 = (max_id + 5 > 5) ? max_id - 1 : max_id + 5;

	for ( int row = 0; row < flow_norm.rows; row++ ) {
		Pixel2* ptr = flow_norm.ptr<Pixel2>(row, 0);
		Pixelc* ptr2 = out_img.ptr<Pixelc>(row, 0);
		for ( int col = 0; col < flow_norm.cols; col++ ) {
			int bin = static_cast<int>((atan2 (ptr->x, ptr->y) * 180 / M_PI + 180) / 360 * 6);
			if (bin == max_id || bin == max_near1 || bin == max_near2) {
				ptr2->x = 0;
				ptr2->y = 0;
				ptr2->z = 0;
			}
			ptr++;
			ptr2++;
		}
	}
}


void fn_dir_color::run_norm_filter (int buffer_size, int buffer_size_filter) {
	cout << "Running color map" << endl;

	VideoWriter* video_output_color = ini_video_output (file_name + "_norm_color_" + to_string(buffer_size) + "_color");
	VideoWriter* video_output_overlay = ini_video_output (file_name + "_norm_color_" + to_string(buffer_size) + "_overlay");

	ini_frame();
	ini_buffer(buffer_size);
	ini_draw_colorwheel ();

	/* ---------- For filtering buffer ---------- */
	// Buffer initialization
	int current_buffer_filter = 0;
	vector<Mat> buffer_filter;
	Mat average_flow_filter = Mat::zeros(height,width,CV_32FC2);
	for ( int i = 0; i < buffer_size_filter; i++ )
	{
		buffer_filter.push_back(Mat::zeros(height,width,CV_32FC2));
	}
	/* ------------------------------------------ */

	for (int framecount = 1; true; ++framecount) {

		if (read_frame()) break;

		calc_FB ();

		Mat out_img;
		Mat out_img_overlay;
		resized_frame.copyTo(out_img);
		resized_frame.copyTo(out_img_overlay);

		/* ---------- For filtering buffer ---------- */
		// Do this before normalize_flow()
		average_flow_filter -= buffer_filter[current_buffer_filter] / static_cast<float>(buffer_size_filter);
		buffer_filter[current_buffer_filter] = flow.clone();
		average_flow_filter += buffer_filter[current_buffer_filter] / static_cast<float>(buffer_size_filter);
		current_buffer_filter++;
		if ( current_buffer_filter >= buffer_size_filter ) current_buffer_filter = 0;
		/* ------------------------------------------ */

		eliminate_std(5);
		normalize_flow();

		update_buffer (buffer_size);

		vector_to_color (average_flow, out_img);

		find_incoming_dir (average_flow_filter, average_flow, out_img);
		drawFrameCount(out_img, framecount);


		addWeighted( out_img, 1.0, out_img_overlay, 1.0, 0.0, out_img_overlay);

		draw_colorwheel (out_img);
		draw_colorwheel (out_img_overlay);
		
		
		imshow ("grid_buoy color map", out_img);
		imshow ("grid_buoy overlay", out_img_overlay);
		video_output_color->write (out_img);
		video_output_overlay->write (out_img_overlay);

		curr_frame.copyTo (prev_frame);
		if ( waitKey(1) == 27) break;

	}

	// clean up
	video_output_color->release();
	video_output_overlay->release();
	destroyAllWindows();

}

void fn_dir_color::run_dir (int buffer_size) {
	cout << "Running direction only color map" << endl;

	VideoWriter* video_output_color = ini_video_output (file_name + "_dir_only_color_" + to_string(buffer_size) + "_color");
	VideoWriter* video_output_overlay = ini_video_output (file_name + "_dir_only_color_" + to_string(buffer_size) + "_overlay");

	ini_frame();
	ini_buffer(buffer_size);
	ini_draw_colorwheel ();

	for (int framecount = 1; true; ++framecount) {

		if (read_frame()) break;

		calc_FB ();

		Mat out_img;
		Mat out_img_overlay;
		resized_frame.copyTo(out_img);
		resized_frame.copyTo(out_img_overlay);

		update_buffer (buffer_size);

		vector_to_dir_color (average_flow, out_img);
		drawFrameCount(out_img, framecount);

		addWeighted( out_img, 0.5, out_img_overlay, 0.5, 0.0, out_img_overlay);	

		draw_colorwheel (out_img);
		draw_colorwheel (out_img_overlay);	
		
		imshow ("grid_buoy color map", out_img);
		imshow ("grid_buoy overlay", out_img_overlay);
		video_output_color->write (out_img);
		video_output_overlay->write (out_img_overlay);

		curr_frame.copyTo (prev_frame);

		if ( waitKey(1) == 27) break;

	}

	// clean up
	video_output_color->release();
	video_output_overlay->release();
	destroyAllWindows();
}