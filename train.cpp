#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "train.h"
#include "dir.h"

using namespace std;
using namespace cv;


vector<TrainedImage> load_training_images(string training_dir) {
	struct dirent *ent;

	vector<TrainedImage> trained_images;

  vector<string> files;
  int dir_ret = getdir(training_dir, files);

  if(dir_ret != 0) {
    perror("Could not open directory");
    return trained_images;
  }

  for(auto f : files) {
	  string img_path(f);
	  if (img_path.compare(".") == 0 || img_path.compare("..") == 0)
	  	continue;

	  trained_images.push_back(TrainedImage(training_dir + "/" + img_path));
  } 

	return trained_images;
}
