#include "Matcher.h"
#include <string>

using namespace std;

vector<SearchResult> get_matches(std::vector<BYTE> &image_data, vector<TrainedImage>& trained_images, FeaturedImage &scene) {
	scene = FeaturedImage(image_data, false);
	return get_matches(trained_images, scene);
}

vector<SearchResult> get_matches(string path, vector<TrainedImage>& trained_images, FeaturedImage &scene) {
	scene = FeaturedImage(path, false);
	return get_matches(trained_images, scene);
}

vector<SearchResult> get_matches(vector<TrainedImage>& trained_images, FeaturedImage &scene) {

	vector<SearchResult> results;

	for (auto ti : trained_images) {

		SearchResult sr = SearchResult(ti);

		sr.feature_matches = get_good_feature_matches(scene, ti);

		if (sr.feature_matches.size() <= 5)
			continue;

		perform_homography(sr, scene);

		if (sr.H.total() == 0)
			continue;

		compute_object_corners(sr);

		if (false && ti.path.find("converter") != std::string::npos)
		{
			std::cout << "Found it " << endl;
			std::cout << "Matches, size: " << sr.feature_matches.size() << endl;
			std::cout << "Rows: " << sr.t.rows << endl;
			std::cout << "Mask total: " << sr.mask.total() << endl;
			std::cout << "Inliers: " << sr.inliers << endl;
			std::cout << "Outliers: " << sr.outliers << endl;
			std::cout << "Ratio: " << sr.inlier_ratio << endl;
		}

		if (sr.match())
			results.push_back(sr);
	}

	scene.gpu_descriptors.release();

	return results;
}

bool SearchResult::match()
{
	if (this->t.rows == 0)
		return false;

	if (this->mask.total() < 3)
		return false;

	if (this->inliers < 20 && (this->inlier_ratio < 0.8))
		return false;

	return true;
}

vector<cv::DMatch> get_good_feature_matches(FeaturedImage &fi, TrainedImage &ti)
{
	cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
	vector<vector<cv::DMatch> > knn_matches;

	cv::cuda::GpuMat s_desc(ti.descriptors);

	matcher->knnMatch(s_desc, fi.gpu_descriptors, knn_matches, 2);

	s_desc.release();

	vector<cv::DMatch> good;

	for (vector<vector<cv::DMatch> >::const_iterator it = knn_matches.begin(); it != knn_matches.end(); ++it) {
		if (it->size() > 1 && (*it)[0].distance / (*it)[1].distance < 0.74) {
			good.push_back((*it)[0]);
		}
	}

	return good;
}

void perform_homography(SearchResult &sr, FeaturedImage &fi)
{
	for (vector<cv::DMatch>::const_iterator it = sr.feature_matches.begin(); it != sr.feature_matches.end(); it++) {
		sr.src_pts.push_back(sr.matching_image.keypoints[it->queryIdx].pt);
		sr.dst_pts.push_back(fi.keypoints[it->trainIdx].pt);
	}

	sr.t = estimateRigidTransform(sr.src_pts, sr.dst_pts, false);

	sr.H = findHomography(sr.src_pts, sr.dst_pts, cv::RANSAC, 5.0, sr.mask);
	
	// Convert to a 2 dimensional box to avoid odd polygons.
	if (sr.t.total()  > 0)
	{
		cv::Mat M(3, 3, cv::DataType<float>::type, 0.0);

		sr.t.row(0).copyTo(M.row(0));
		sr.t.row(1).copyTo(M.row(1));

		M.at<float>(2, 2) = 1;

		sr.H = M;
	}

	sr.inliers = cv::countNonZero(sr.mask);
	sr.outliers = (int)(sr.mask.total()) - cv::countNonZero(sr.mask);
	sr.inlier_ratio = (float)sr.inliers / (float)sr.mask.total();
}

void compute_object_corners(SearchResult &sr)
{
	std::vector<cv::Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); 
	obj_corners[1] = cvPoint(sr.matching_image.image.cols, 0);
	obj_corners[2] = cvPoint(sr.matching_image.image.cols, sr.matching_image.image.rows); 
	obj_corners[3] = cvPoint(0, sr.matching_image.image.rows);
	sr.scene_corners = std::vector<cv::Point2f>(4);
	perspectiveTransform(obj_corners, sr.scene_corners, sr.H);
}