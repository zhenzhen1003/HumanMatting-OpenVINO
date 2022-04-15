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

public:
    explicit OpenVINO(const std::string& network_xml_filepath);   //构造函数
    void predict();                   //视频推理
    void predict(cv::Mat& image);     //图片推理
    void setVideoCapture(cv::VideoCapture *reader, const std::string& path);   //设置视频读取指针


    /// <summary>
    /// 推理请求声明
    /// </summary>
    InferenceEngine::InferRequest infer_request;
    InferenceEngine::InferRequest infer_request1;
    InferenceEngine::InferRequest infer_request2;
    InferenceEngine::InferRequest infer_request3;

    //输出的alpha图
    cv::Mat alpha = cv::Mat::zeros(1080, 1920, CV_8UC3);
    //视频读取句柄
    cv::VideoCapture *reader;
    //视频写入句柄
    cv::VideoWriter* writer;
    //存储从视频读取的一帧
    cv::Mat frame;

public:
    //将Mat图像数据转换为Openvino可以处理的数据
    inline void preprocess(cv::Mat& image, InferenceEngine::Blob::Ptr& blob);

    //读取的视频的大小、帧率、总帧数
    int input_height = 1080;
    int input_width = 1920;
    int input_fps = 25;
    int frame_num;
};

#endif // OPENVINO_H
