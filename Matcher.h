#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <vector>

#include "train.h"

using namespace std;

class SearchResult {

public:

	TrainedImage matching_image;

	string destination;

	vector<cv::Point2f> src_pts, dst_pts;
	
	int inliers;
	int outliers;
	float inlier_ratio;

	cv::UMat image;

	vector<cv::DMatch> feature_matches;

	cv::Mat t;
	cv::Mat mask;
	cv::Mat H;
	std::vector<cv::Point2f> scene_corners;

	// Need matchesMask and all homogrpahy parameters

	bool match();
	
	SearchResult() {  } 
	SearchResult(TrainedImage matching_image) { this->matching_image = matching_image; }
};

vector<SearchResult> get_matches(vector<TrainedImage>& trained_images, FeaturedImage &scene);
vector<SearchResult> get_matches(string path, vector<TrainedImage>& trained_images, FeaturedImage &scene);
vector<SearchResult> get_matches(vector<BYTE> &image_data, vector<TrainedImage>& trained_images, FeaturedImage &scene);
vector<cv::DMatch> get_good_feature_matches(FeaturedImage &fi, TrainedImage &ti);
void perform_homography(SearchResult &sr, FeaturedImage &fi);
void compute_object_corners(SearchResult &sr);