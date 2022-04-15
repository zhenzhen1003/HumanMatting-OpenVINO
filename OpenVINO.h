#ifndef OPENVINO_H
#define OPENVINO_H

#include <string>
#include <vector>
#include <fstream>
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>



class OpenVINO//: public ImageFilter
{
protected:
    InferenceEngine::ExecutableNetwork executable_network;
    InferenceEngine::InferRequest infer_request;

public:
    explicit OpenVINO(const std::string& network_xml_filepath);
    void predict();
    void predict(cv::Mat& image);
    void setVideoCapture(cv::VideoCapture *reader);
    cv::Mat alpha = cv::Mat::zeros(1080, 1920, CV_8UC3);
    cv::VideoCapture *reader;
    cv::VideoWriter* writer;
    cv::Mat frame;

public:
    inline void preprocess(cv::Mat& image, InferenceEngine::Blob::Ptr& blob);

    int input_height = 1080;
    int input_width = 1920;
    int input_fps = 25;
    int frame_num;
};

#endif // OPENVINO_H
