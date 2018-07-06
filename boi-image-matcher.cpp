#include <opencv2/core/core.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <thread>

#include "train.h"
#include "Matcher.h"
#include "zmq.hpp"
#include "base64.h"

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

#include <sstream>
#include <mutex>

#include <stdint.h>
typedef uint8_t           BYTE;
#include <unistd.h>

//void sleep(unsigned milliseconds)
//{
//	usleep(milliseconds * 1000); // takes microseconds
//}

using namespace std;

#define TRAINING_IMAGES "/home/gnychis/data/boi-training"

static const int NUM_THREADS = 4;

std::mutex mtx;

bool SHOW_IMAGE = false;
bool THREAD_SAFE = true;
bool READ_AS_BYTES = true;


std::vector<BYTE> matToBytes(cv::Mat image)
{
	vector<BYTE> v_char;
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			v_char.push_back(*(uchar*)(image.data + i*image.step + j));

		}
	}
	return v_char;
}

void *worker_routine(void *arg, vector<TrainedImage> &trained_images)
{
	zmq::context_t *context = (zmq::context_t *) arg;

	zmq::socket_t socket(*context, ZMQ_DEALER);
	socket.connect("inproc://backend");

	std::cerr << "This is a thread" << std::endl;

	while (true) {		

		// Receive the identity and copy it back in to another variable to address the response.
		zmq::message_t identity;
		zmq::message_t copied_id;
		socket.recv(&identity);
		copied_id.copy(&identity);

		zmq::message_t request;
		socket.recv(&request);
		
		std::string identityStr = std::string(static_cast<char *>(identity.data()), identity.size());
		std::string requestStr = std::string(static_cast<char *>(request.data()), request.size());
		
		//////////////////////
		rapidjson::Document jsonRequest;
		jsonRequest.Parse(requestStr.c_str());

		rapidjson::Value& tweet_id_json = jsonRequest["tweet_id"];

		int64_t tweet_id = tweet_id_json.GetInt64();
		std::string tweet_image_b64_str = jsonRequest["image"].GetString();

		std::cerr << "Received from client: " + identityStr + ", Tweet: " + std::to_string(tweet_id) << std::endl;

		///////////////////////
		std::vector<BYTE> myData;
		std::vector<BYTE> decodedData = base64_decode(tweet_image_b64_str);

		//////////////////////
		if(!THREAD_SAFE)
			mtx.lock();
		FeaturedImage scene;

		///////////////////
		std::ostringstream oss;
		oss << tweet_id;

		rapidjson::Document rj_response;
		rj_response.SetObject();
		rapidjson::Value matches(rapidjson::kArrayType);
		rapidjson::Document::AllocatorType& allocator = rj_response.GetAllocator();

		rapidjson::Value rj_tweet_id(oss.str().c_str(), rj_response.GetAllocator());
		rj_response.AddMember("tweet_id", rj_tweet_id, allocator);

		vector<SearchResult> results;
		for (auto sr : results) {

			if (!sr.match())
				continue;

			rapidjson::Value match(rapidjson::kObjectType);
			rapidjson::Value rj_path(sr.matching_image.path.c_str(), rj_response.GetAllocator());
			match.AddMember("path", rj_path, allocator);

			std:string found(oss.str());
			found += "|";

			found += sr.matching_image.path + ";";

			//cout << sr.matching_image.path << endl;

			cv::Mat img_matches;
			cv::Mat empty;
			std::vector< cv::DMatch > emptyVec;
			//drawMatches(sr.matching_image.image, sr.matching_image.keypoints, scene.image, scene.keypoints,
			//	sr.feature_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
			//	std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			drawMatches(empty, sr.matching_image.keypoints, scene.image, scene.keypoints,
				emptyVec, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
				std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			//-- Draw lines between the corners (the mapped object in the scene - image_2 )
			line(img_matches, sr.scene_corners[0], sr.scene_corners[1], cv::Scalar(0, 255, 0), 4);
			line(img_matches, sr.scene_corners[1], sr.scene_corners[2], cv::Scalar(0, 255, 0), 4);
			line(img_matches, sr.scene_corners[2], sr.scene_corners[3], cv::Scalar(0, 255, 0), 4);
			line(img_matches, sr.scene_corners[3], sr.scene_corners[0], cv::Scalar(0, 255, 0), 4);
			//-- Show detected matches

			if (SHOW_IMAGE) {

				imshow("Good Matches & Object detection", img_matches);
				cv::waitKey();
			}

			std::vector<uchar> buff;
			cv::imencode(".png", img_matches, buff);
			std::string boxed_img_base64 = base64_encode(&buff[0], buff.size());
			rapidjson::Value rj_boxed_img(boxed_img_base64.c_str(), rj_response.GetAllocator());
			match.AddMember("boxed_image", rj_boxed_img, allocator);

			matches.PushBack(match, allocator);
		}

		rj_response.AddMember("matches", matches, allocator);

		rapidjson::StringBuffer buffer;
		rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
		rj_response.Accept(writer);
		std::string rj_response_str = buffer.GetString();

		// Send the ID back for the address
		socket.send(copied_id, ZMQ_SNDMORE);
		zmq::message_t response(rj_response_str.length());
		memcpy(response.data(), rj_response_str.c_str(), rj_response_str.length());

		if (!socket.send(response))
			std::cout << "Failed to send " << tweet_id << endl;
		else
			cout << "Sent: " << tweet_id << endl;

		if(!THREAD_SAFE)
			mtx.unlock();

		sleep(1);
	}

	return NULL;
}

int main(int argc, char** argv)
{
	// Load the training images.
	vector<TrainedImage> trained_images;
	FeaturedImage scene;
	trained_images = load_training_images(TRAINING_IMAGES);

	std::cout << "Arguments: " << sizeof(argv) << endl;

	if (argc <= 1)
	{
		zmq::context_t context(1);
		zmq::socket_t frontend(context, ZMQ_ROUTER);
		zmq::socket_t backend(context, ZMQ_DEALER);

		frontend.bind("tcp://*:5555");
		backend.bind("inproc://backend");

		// Launch pool of worker threads
		std::thread t[NUM_THREADS];
		for (int thread_nbr = 0; thread_nbr != NUM_THREADS; thread_nbr++) {
			t[thread_nbr] = std::thread(worker_routine, &context, std::ref(trained_images));
		}

		zmq::proxy(frontend, backend, NULL);

		for (int thread_nbr = 0; thread_nbr != NUM_THREADS; thread_nbr++) {
			t[thread_nbr].join();
		}
	}
	else
	{
		// Get a result
		vector<SearchResult> results;
		if (!READ_AS_BYTES)
		{
			results = get_matches(argv[1], trained_images, scene);
		}
		else
		{
			std::ifstream file(argv[1], std::ios::binary);
			file.unsetf(std::ios::skipws);
			std::streampos fileSize;

			file.seekg(0, std::ios::end);
			fileSize = file.tellg();
			file.seekg(0, std::ios::beg);

			// reserve capacity
			std::vector<BYTE> vec;
			vec.reserve(fileSize);

			// read the data:
			vec.insert(vec.begin(),
				std::istream_iterator<BYTE>(file),
				std::istream_iterator<BYTE>());

			std::cout << "Getting as a vector" << endl;
			results = get_matches(vec, trained_images, scene);
		}
		for (auto sr : results) {
			std::cout << sr.matching_image.path << endl;

			cv::Mat img_matches;
			drawMatches(sr.matching_image.image, sr.matching_image.keypoints, scene.image, scene.keypoints,
				sr.feature_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
				std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			//-- Draw lines between the corners (the mapped object in the scene - image_2 )
			line(img_matches, sr.scene_corners[0] + cv::Point2f(sr.matching_image.image.cols, 0), sr.scene_corners[1] + cv::Point2f(sr.matching_image.image.cols, 0), cv::Scalar(0, 255, 0), 4);
			line(img_matches, sr.scene_corners[1] + cv::Point2f(sr.matching_image.image.cols, 0), sr.scene_corners[2] + cv::Point2f(sr.matching_image.image.cols, 0), cv::Scalar(0, 255, 0), 4);
			line(img_matches, sr.scene_corners[2] + cv::Point2f(sr.matching_image.image.cols, 0), sr.scene_corners[3] + cv::Point2f(sr.matching_image.image.cols, 0), cv::Scalar(0, 255, 0), 4);
			line(img_matches, sr.scene_corners[3] + cv::Point2f(sr.matching_image.image.cols, 0), sr.scene_corners[0] + cv::Point2f(sr.matching_image.image.cols, 0), cv::Scalar(0, 255, 0), 4);
			//-- Show detected matches

			if (SHOW_IMAGE)
			{
				imshow("Good Matches & Object detection", img_matches);
				cv::waitKey();
			}
		}
	}

	return 0;
}


/*
int main() {
//  Prepare our context and socket
zmq::context_t context(1);
zmq::socket_t socket(context, ZMQ_PAIR);
socket.bind("tcp://*:5555");

// forever loop
while (true) {
zmq::message_t request;

//  Wait for next request from client
socket.recv(&request);
std::string replyMessage = std::string(static_cast<char *>(request.data()), request.size());
//        std::string replyMessage = std::string((request.data())., request.size());
// Print out received message
std::cout << "Received from client: " + replyMessage << std::endl;

//  See the gradual sending/replying from client
sleep(1);

//  Send reply back to client
std::string msgToClient("greeting from C++");
zmq::message_t reply(msgToClient.size());
memcpy((void *)reply.data(), (msgToClient.c_str()), msgToClient.size());
socket.send(reply);
}
return 0;
}*/
