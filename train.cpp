#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "train.h"

using namespace std;
using namespace cv;


vector<TrainedImage> load_training_images(string training_dir) {
	DIR *dir;
	struct dirent *ent;

	vector<TrainedImage> trained_images;

	if ((dir = opendir(training_dir.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {

			string img_path(ent->d_name);
			if (img_path.compare(".") == 0 || img_path.compare("..") == 0)
				continue;

			trained_images.push_back(TrainedImage(training_dir + img_path));
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		throw;
	}

	return trained_images;
}
