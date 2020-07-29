#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "fn_timeline.hpp"

using namespace std;

void mouse_callback(int event, int x, int y, int, void *userdata)
{
    if (event == EVENT_LBUTTONDOWN) {
		cout << x << " " << y << endl;
		pair<vector<pair<Pixel2, Pixel2>>, Pixel2*> *p = static_cast<pair<vector<pair<Pixel2, Pixel2>>, Pixel2*>*>(userdata);
		if (p->second == nullptr) {
			p->second = new Pixel2(x,y);
		} 
		else {
			p->first.push_back(make_pair(*(p->second), Pixel2(x,y)));
			p->second = nullptr;
		}

    }
}

fn_timeline::fn_timeline (string file_name, 
						  string _outfile_dir,
						  int _height,
						  int _vnum,
						  int _born_period,
						  int _lifespan): method(file_name, _outfile_dir, _height) {
	vnum = _vnum;
	born_period = _born_period;
	lifespan = _lifespan;
}

void fn_timeline::run (bool isNorm) {
	cout << "Running timeline" << endl;

	ini_frame();

	imshow ("click start and end", resized_frame);

	pair<vector<pair<Pixel2, Pixel2>>, Pixel2*> vec_and_pixel;

	setMouseCallback("click start and end", mouse_callback, &vec_and_pixel);

	while (vec_and_pixel.first.size() == 0) {
		waitKey();
	}

	for (auto pixels : vec_and_pixel.first) {
		start_end.push_back (make_pair(pixels.first, pixels.second));
		add_timeline (pixels.first, pixels.second, vnum, lifespan);
	}

	string n_str = isNorm? "norm_" : "";

	VideoWriter* video_output = ini_video_output (file_name +  "_timelines_" + n_str
		+ to_string(static_cast<int>(start_end[0].first.x)) + "_" 
		+ to_string(static_cast<int>(start_end[0].first.y)) + "_"
		+ to_string(static_cast<int>(start_end[0].second.x)) + "_" 
		+ to_string(static_cast<int>(start_end[0].second.y)) + "_"
		+ to_string(vnum));

	for (int framecount = 1; true; ++framecount) {

		cout << "Frame " << framecount << endl;

		if (read_frame()) break;

		Mat out_img;
		resized_frame.copyTo (out_img);

		if (born_period != 0 && framecount%born_period == 0) {
			for (auto pixels : start_end) {
				add_timeline (pixels.first, pixels.second, vnum, framecount + lifespan);
			}
		}

		for (auto begin = timelines.begin(); begin != timelines.end();) {
			if (begin->die_at != 0 && begin->die_at < framecount){
				timelines.erase(begin);
			}
			else {
				begin->runLK (prev_frame, curr_frame, out_img, isNorm);
				++begin;
			}
		}

		// Draw gray lines as initial position of the timelines
		for (auto pixels : start_end) {
			line(out_img,Point(pixels.first.x,pixels.first.y),Point(pixels.second.x,pixels.second.y),CV_RGB(50,50,50),2,8,0);
		}


		drawFrameCount(out_img, framecount);
		
		imshow ("timelines", out_img);
		video_output->write (out_img);

		curr_frame.copyTo (prev_frame);
		if ( waitKey(1) == 27) break;

	}

	// clean up
	video_output->release();
	destroyAllWindows();

}

void fn_timeline::add_timeline (Pixel2 start, Pixel2 end, int vertices_count, int die_at) {
	timelines.push_back(timeline(start, end, vertices_count, die_at));
}

timeline::timeline (Pixel2 start, Pixel2 end, int vertices_count, int _die_at) {

	die_at = _die_at;
	
	// define the distance between each vertices
	float diffX = (end.x - start.x) / vertices_count;
	float diffY = (end.y - start.y) / vertices_count;

	// create and push Pixel2 points
	for (int i = 0; i <= vertices_count; ++i) {
		vertices.push_back(Pixel2(start.x + diffX * i, start.y + diffY * i));
	}

}


void timeline::runLK(Mat u_prev, Mat u_curr, Mat& out_img, bool isNorm) {

	// return status values of calcOpticalFlowPyrLK
	vector<uchar> status;
	vector<float> err;


	// output locations of vertices
	vector<Pixel2> vertices_next;

	// run LK for all vertices
	calcOpticalFlowPyrLK(u_prev, u_curr, vertices, 
						 vertices_next, status, err, 
						 Size(50,50),3, 
						 TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.1), 
						 10, 1e-4 );



	// eliminate any large movement
	for ( int i = 0; i < (int)vertices_next.size(); i++) {
		if ( abs(vertices[i].x - vertices_next[i].x) > 20
			|| abs(vertices[i].y - vertices_next[i].y) > 20 ) {
				vertices_next[i] = vertices[i];
			}
		
		if (isNorm) {
			float x = vertices_next[i].x - vertices[i].x;
			float y = vertices_next[i].y - vertices[i].y;
			
			float theta = atan2 (y, x);

			// float dt = 1;
			float dt = 0.1;
		
			vertices_next[i].x = vertices[i].x + cos(theta) * dt;
			vertices_next[i].y = vertices[i].y + sin(theta) * dt;
		} 
	}

    // Calculate average
    /*
    Pixel2 mean = Pixel2(0,0);
    for ( int i = 0; i < (int)vertices_next.size(); i++) {
        mean.x += vertices_next[i].x - vertices[i].x;
        mean.y += vertices_next[i].y - vertices[i].y;
    }
    mean.x = mean.x / vertices_next.size();
    mean.y = mean.y / vertices_next.size();

    for ( int i = 0; i < (int)vertices_next.size(); i++) {
        vertices_next[i].x -= mean.x;
        vertices_next[i].y -= mean.y;
    }
    */
	
	// copy the result for the next frame
	vertices = vertices_next;

	/*
	// delete out of bound vertices
	for ( int i = 0; i < (int)vertices.size(); i++) {
		// if vertex is not in the image
		//printf("%d %f \n", YDIM, vertices.at(i).y);
		if (vertices.at(i).x <= 0 || vertices.at(i).x >= XDIM || vertices.at(i).y <= 0 || vertices.at(i).y >= YDIM) {
			vertices.erase(vertices.begin(), vertices.begin() + i);
		}
	}
	*/

	// draw edges
	circle(out_img,Point(vertices[0].x,vertices[0].y),4,CV_RGB(0,0,100),-1,8,0);
	for ( int i = 0; i < (int)vertices.size() - 1; i++ ) {
		line(out_img,Point(vertices[i].x,vertices[i].y),Point(vertices[i+1].x,vertices[i+1].y),CV_RGB(100,0,0),2,8,0);
		circle(out_img,Point(vertices[i+1].x,vertices[i+1].y),4,CV_RGB(0,0,100),-1,8,0);
	}
}