#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "featuredimg.h"
#include "Matcher.h"

#include <mutex>
std::mutex match_mtx;

using namespace std;
using namespace cv;

FeaturedImage::FeaturedImage() {

}

FeaturedImage::FeaturedImage(std::vector<BYTE> &img_data, bool release_gpu_descriptors, int edge_threshold) {
	this->feature(img_data, release_gpu_descriptors, edge_threshold);
}

FeaturedImage::FeaturedImage(string path, bool release_gpu_descriptors, int edge_threshold) {
	this->feature(path, release_gpu_descriptors, edge_threshold);
}

void FeaturedImage::feature(std::vector<BYTE> &img_data, bool release_gpu_descriptors, int edge_threshold) {
	imdecode(img_data, CV_LOAD_IMAGE_COLOR).copyTo(this->image);
	this->compute_features(release_gpu_descriptors, edge_threshold);
}

void FeaturedImage::feature(string path, bool release_gpu_descriptors, int edge_threshold) {
	this->path = path;
	imread(this->path).copyTo(this->image);
	this->compute_features(release_gpu_descriptors, edge_threshold);
}

void FeaturedImage::compute_features(bool release_gpu_descriptors, int edge_threshold) {
  
  cv::Mat hsv_image, mask, frame_threshold, rgb_image;
  cv::cvtColor(this->image, hsv_image, cv::COLOR_BGR2HSV);
  cv::inRange(hsv_image, Scalar(0,255,40), Scalar(0,255,255), frame_threshold);

  for(int j=0; j<hsv_image.rows; j++)
  {
    for(int i=0; i<hsv_image.cols; i++)
    {
      if(frame_threshold.at<uchar>(j,i) == 255) {
        hsv_image.at<Vec3b>(j,i)[0] = 80;
      }
    }
  }

  cv::cvtColor(hsv_image, rgb_image, cv::COLOR_HSV2RGB);
  cv::cvtColor(rgb_image, this->image, cv::COLOR_RGB2GRAY); 
	GaussianBlur(this->image, this->image, Size(3, 3), 0);

	this->gpu_image.upload(this->image);

	Ptr<cuda::ORB> orb = cuda::ORB::create(10000, 1.2F, 8, edge_threshold);

	match_mtx.lock();
	try {
		orb->detectAndCompute(this->gpu_image, cuda::GpuMat(), this->keypoints, this->gpu_descriptors);
	}
	catch (cv::Exception& e)
	{
		std::cout << "Exception";
	}
	match_mtx.unlock();

	this->gpu_descriptors.download(this->descriptors);

	if (release_gpu_descriptors)
		this->gpu_descriptors.release();

	this->gpu_image.release();
}
