#ifndef ROTATE_INCLUDE_ROTATED_H_
#define ROTATE_INCLUDE_ROTATED_H_
#include <memory>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <dirent.h>
#include "cuda_runtime_api.h"
#include "NvInfer.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;
using namespace nvinfer1;

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float>maskdata;
    cv::Mat mask;
};

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class Rotatation {
public:
    void loadEngine(const std::string& path);
    cv::Mat preprocessImg(cv::Mat& img, const int& input_w, const int& input_h, int& padw, int& padh);
    std::vector<std::string> listJpgFiles(const std::string& directory) ;
    void totalInference(const std::string& directory);

    void LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
	bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color);

private:
    Logger logger_;
    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* context_ = nullptr;

    float CONF_THRESHOLD = 0.3;
    float NMS_THRESHOLD = 0.3;

    const int BATCH_SIZE = 1;
    const int CHANNELS = 3;
    const int INPUT_H = 1024;
    const int INPUT_W = 1024;

    const int OUTPUT0_BOXES = 21504;
    const int OUTPUT0_ELEMENT = 20;
    const int CLASSES = 15;

    const char* images_ = "images";
    const char* output0_ = "output0";

    std::vector<std::string> class_names{
         "plane",  "ship", "storage tank", 
		"baseball diamond",  "tennis court",   "basketball court", 
		"ground track field", "harbor",  "bridge",  
		"large vehicle",  "small vehicle",  "helicopter",
		"roundabout",  "soccer ball field",  "swimming pool"
    };

};


#endif //ROTATE_INCLUDE_ROTATED_H_