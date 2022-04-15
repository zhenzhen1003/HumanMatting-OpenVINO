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
    explicit OpenVINO(const std::string& network_xml_filepath);   //���캯��
    void predict();                   //��Ƶ����
    void predict(cv::Mat& image);     //ͼƬ����
    void setVideoCapture(cv::VideoCapture *reader, const std::string& path);   //������Ƶ��ȡָ��


    /// <summary>
    /// ������������
    /// </summary>
    InferenceEngine::InferRequest infer_request;
    InferenceEngine::InferRequest infer_request1;
    InferenceEngine::InferRequest infer_request2;
    InferenceEngine::InferRequest infer_request3;

    //�����alphaͼ
    cv::Mat alpha = cv::Mat::zeros(1080, 1920, CV_8UC3);
    //��Ƶ��ȡ���
    cv::VideoCapture *reader;
    //��Ƶд����
    cv::VideoWriter* writer;
    //�洢����Ƶ��ȡ��һ֡
    cv::Mat frame;

public:
    //��Matͼ������ת��ΪOpenvino���Դ��������
    inline void preprocess(cv::Mat& image, InferenceEngine::Blob::Ptr& blob);

    //��ȡ����Ƶ�Ĵ�С��֡�ʡ���֡��
    int input_height = 1080;
    int input_width = 1920;
    int input_fps = 25;
    int frame_num;
};

#endif // OPENVINO_H
