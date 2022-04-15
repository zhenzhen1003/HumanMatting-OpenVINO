#include <iostream>

#include <OpenVINO.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <windows.h>
#include "cmdline.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    // Parse command line arguments
    cmdline::parser parser;
    parser.add<string>("path", 'p', "Path to file", true, "");
    parser.add("help", 'h', "Print help");
    parser.add("video", 'v', "video file");
    parser.add("image", 'i', "image file");
    parser.parse_check(argc, argv);

    //读取参数
    string path = parser.get<string>("path");
    bool isVideo = parser.exist("video");
    bool isImage = parser.exist("image");


    try {
        OpenVINO* openvino = new OpenVINO("./model/FP16/robust.xml");

        if (isImage)
        {
            cv::Mat image = cv::imread(path);
            DWORD start_times = GetTickCount();
            openvino->predict(image);
            DWORD end_times = GetTickCount();
            cout << "图像处理时间" << end_times - start_times << "ms!" << endl;
        }
        if (isVideo)
        {
            cv::VideoCapture* reader = new cv::VideoCapture(path);
            openvino->setVideoCapture(reader);
            DWORD start_times = GetTickCount();
            openvino->predict();
            DWORD end_times = GetTickCount();
            cout << "视频处理平均帧率" << 1000 / ((end_times - start_times) / openvino->frame_num) << "fps!" << endl;
        }

    }
    // -*- error -*- //

    catch (std::bad_alloc& e) {
        std::cerr << "BAD ALLOC Exception : " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    catch (...) {
        std::cerr << "unknown exception" << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}
