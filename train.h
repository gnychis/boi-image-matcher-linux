#pragma once

#include "featuredimg.h"

class TrainedImage : public FeaturedImage
{
public:
	TrainedImage(string path) : FeaturedImage(path, true, 0) { }
	TrainedImage() : FeaturedImage() { }
};

vector<TrainedImage> load_training_images(string training_dir);