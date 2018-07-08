#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
typedef unsigned char       BYTE;

class FeaturedImage {

public:
	string path;
	vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	cv::UMat image;
	cv::cuda::GpuMat gpu_image;
	cv::cuda::GpuMat gpu_descriptors;

	FeaturedImage(std::vector<BYTE> &data, bool release_gpu_descriptors = true, int edge_threshold = 32);
	FeaturedImage(string path, bool release_gpu_descriptors = true, int edge_threshold = 32);

	FeaturedImage();

	void feature(std::vector<BYTE> &data, bool release_gpu_descriptors = true, int edge_threshold = 32);
	void feature(string path, bool release_gpu_descriptors = true, int edge_threshold = 32);

private:
	void compute_features(bool release_gpu_descriptors, int edge_threshold, int mask_val = 255, int replace=0);
};
