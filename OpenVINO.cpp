#include "OpenVINO.h"
#include <windows.h>

using namespace std;

//并行推理的流数
const int N = 4;

OpenVINO::OpenVINO(const std::string& network_xml_filepath)
{
    InferenceEngine::Core ie;
    std::vector<std::string> device = ie.GetAvailableDevices();
    std::map<std::string, std::string> config = {{ InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "4"} };
    std::string device_name("AUTO");
    InferenceEngine::CNNNetwork model = ie.ReadNetwork(network_xml_filepath);
    model.setBatchSize(1);
    InferenceEngine::InputInfo::Ptr input_info = model.getInputsInfo().begin()->second;
    input_info->setPrecision(InferenceEngine::Precision::U8);
    for(auto& item : model.getOutputsInfo())
    {
        auto&  output_info = item.second;
        output_info->setPrecision(InferenceEngine::Precision::FP32);
    }
    ie.SetConfig({ { CONFIG_KEY(GPU_THROUGHPUT_STREAMS),std::to_string(N) } }, "GPU");
    ie.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS),std::to_string(N) } }, "CPU");

    executable_network = ie.LoadNetwork(model, device_name);

    infer_request = executable_network.CreateInferRequest();
    infer_request1 = executable_network.CreateInferRequest();
    infer_request2 = executable_network.CreateInferRequest();
    infer_request3 = executable_network.CreateInferRequest();
}

void OpenVINO::setVideoCapture(cv::VideoCapture *reader, const std::string& path)
{
    this->reader = reader;
    input_width = this->reader->get(cv::CAP_PROP_FRAME_WIDTH);
    input_height = this->reader->get(cv::CAP_PROP_FRAME_HEIGHT);
    input_fps = this->reader->get(cv::CAP_PROP_FPS);
    frame_num = this->reader->get(cv::CAP_PROP_FRAME_COUNT);
    writer = new cv::VideoWriter("test.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), input_fps, cv::Size(input_width, input_height));
}

void OpenVINO::preprocess(cv::Mat& image, InferenceEngine::Blob::Ptr& blob)
{
    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    auto mblobHolder = mblob->wmap();
    uint8_t* blob_data = mblobHolder.as<uint8_t*>();
    cv::resize(image, image, cv::Size(1920, 1080));

    for(int c = 0; c < 3; c++)
    {
        for(int  h = 0; h < 1080; h++)
        {
            for(int w = 0; w < 1920; w++)
            {
                blob_data[c * 1920 * 1080 + h * 1920 + w] =
                    (uint8_t)image.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}


void OpenVINO::predict()
{
    cv::Mat output;
    cv::Mat input;
    bool flag1 = false;
    bool flag2 = false;
    bool flag3 = false;
    while (reader->read(frame))
    {
        flag1 = false;
        flag2 = false;
        flag3 = false;
        cv::resize(frame, input, cv::Size(1920, 1080));
        static auto inputBlob = infer_request.GetBlob("src");
        preprocess(input, inputBlob);
        infer_request.StartAsync();

        if (reader->read(frame))
        {
            cv::resize(frame, input, cv::Size(1920, 1080));
            static auto inputBlob1 = infer_request1.GetBlob("src");
            preprocess(input, inputBlob1);
            infer_request1.StartAsync();
            flag1 = true;
        }

        if (reader->read(frame))
        {
            cv::resize(frame, input, cv::Size(1920, 1080));
            static auto inputBlob2 = infer_request2.GetBlob("src");
            preprocess(input, inputBlob2);
            infer_request2.StartAsync();
            flag2 = true;
        }


        if (reader->read(frame))
        {
            cv::resize(frame, input, cv::Size(1920, 1080));
            static auto inputBlob3 = infer_request3.GetBlob("src");
            preprocess(input, inputBlob3);
            infer_request3.StartAsync();
            flag3 = true;
        }

        infer_request.Wait();
        const InferenceEngine::Blob::Ptr dis_pred_blob = infer_request.GetBlob("pha");
        auto mdis_pred = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob);
        auto mdis_pred_holder = mdis_pred->rmap();
        const float* dis_pred = mdis_pred_holder.as<const float*>();
        int index = 0;
        unsigned char temp = 0;
        for (int h = 0; h < 1080; ++h) {
            for (int w = 0; w < 1920; ++w) {
                temp = (unsigned char)(dis_pred[index] * 255.0f);
                alpha.at<cv::Vec3b>(h, w) = { temp, temp, temp }; // R,G,B
                index++;
            }
        }
        cv::resize(alpha, output, cv::Size(input_width, input_height));
        writer->write(output);

        if (flag1)
        {
            infer_request1.Wait();
            const InferenceEngine::Blob::Ptr dis_pred_blob1 = infer_request1.GetBlob("pha");
            auto mdis_pred1 = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob1);
            auto mdis_pred_holder1 = mdis_pred1->rmap();
            const float* dis_pred1 = mdis_pred_holder1.as<const float*>();
            index = 0;
            temp = 0;
            for (int h = 0; h < 1080; ++h) {
                for (int w = 0; w < 1920; ++w) {
                    temp = (unsigned char)(dis_pred1[index] * 255.0f);
                    alpha.at<cv::Vec3b>(h, w) = { temp, temp, temp }; // R,G,B
                    index++;
                }
            }
            cv::resize(alpha, output, cv::Size(input_width, input_height));
            writer->write(output);
        }

        if (flag2)
        {
            infer_request2.Wait();
            const InferenceEngine::Blob::Ptr dis_pred_blob2 = infer_request2.GetBlob("pha");
            auto mdis_pred2 = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob2);
            auto mdis_pred_holder2 = mdis_pred2->rmap();
            const float* dis_pred2 = mdis_pred_holder2.as<const float*>();
            index = 0;
            temp = 0;
            for (int h = 0; h < 1080; ++h) {
                for (int w = 0; w < 1920; ++w) {
                    temp = (unsigned char)(dis_pred2[index] * 255.0f);
                    alpha.at<cv::Vec3b>(h, w) = { temp, temp, temp }; // R,G,B
                    index++;
                }
            }
            cv::resize(alpha, output, cv::Size(input_width, input_height));
            writer->write(output);
        }

        if (flag3)
        {
            infer_request3.Wait();
            const InferenceEngine::Blob::Ptr dis_pred_blob3 = infer_request3.GetBlob("pha");
            auto mdis_pred3 = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob3);
            auto mdis_pred_holder3 = mdis_pred3->rmap();
            const float* dis_pred3 = mdis_pred_holder3.as<const float*>();
            index = 0;
            temp = 0;
            for (int h = 0; h < 1080; ++h) {
                for (int w = 0; w < 1920; ++w) {
                    temp = (unsigned char)(dis_pred3[index] * 255.0f);
                    alpha.at<cv::Vec3b>(h, w) = { temp, temp, temp }; // R,G,B
                    index++;
                }
            }
            cv::resize(alpha, output, cv::Size(input_width, input_height));
            writer->write(output);
        }
    }
    reader->release();
    writer->release();
}



void OpenVINO::predict(cv::Mat &image)
{
    int width = image.cols;
    int height = image.cols;
    cv::resize(image, image, cv::Size(1920,1080));
    static auto inputBlob = infer_request.GetBlob("src");
    preprocess(image, inputBlob);
    infer_request.Infer();
    const InferenceEngine::Blob::Ptr dis_pred_blob = infer_request.GetBlob("pha");
    auto mdis_pred = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob);
    auto mdis_pred_holder = mdis_pred->rmap();
    const float* dis_pred = mdis_pred_holder.as<const float*>();
    int index = 0;
    unsigned char temp = 0;
    for (int h = 0; h < 1080; ++h) {
        for (int w = 0; w < 1920; ++w) {
            temp = (unsigned char)(dis_pred[index] * 255.0f);
            alpha.at<cv::Vec3b>(h, w) = { temp, temp, temp }; // R,G,B
            index++;
        }
    }
    cv::resize(alpha, alpha, cv::Size(width, height));
    cv::imwrite("frame.png", alpha);
}



/*
void OpenVINO::predict()
{
    cv::Mat output;
    cv::Mat input;
    bool flag = false;
    bool flag1 = false;
    bool flag2 = false;
    bool flag3 = false;
    bool flag4 = false;
    bool flag5 = false;
    bool flag6 = false;
    bool flag7 = false;
    while (reader->read(frame))
    {
        flag1 = false;
        flag2 = false;
        flag3 = false;
        flag4 = false;
        flag5 = false;
        flag6 = false;
        flag7 = false;
        cv::resize(frame, input, cv::Size(1920, 1080));
        static auto inputBlob = infer_request.GetBlob("src");
        preprocess(input, inputBlob);
        infer_request.StartAsync();

        if (reader->read(frame))
        {
            cv::resize(frame, input, cv::Size(1920, 1080));
            static auto inputBlob1 = infer_request1.GetBlob("src");
            preprocess(input, inputBlob1);
            infer_request1.StartAsync();
            flag1 = true;
        }

        if (reader->read(frame))
        {
            cv::resize(frame, input, cv::Size(1920, 1080));
            static auto inputBlob2 = infer_request2.GetBlob("src");
            preprocess(input, inputBlob2);
            infer_request2.StartAsync();
            flag2 = true;
        }


        if (reader->read(frame))
        {
            cv::resize(frame, input, cv::Size(1920, 1080));
            static auto inputBlob3 = infer_request3.GetBlob("src");
            preprocess(input, inputBlob3);
            infer_request3.StartAsync();
            flag3 = true;
        }

        if (reader->read(frame))
        {
            cv::resize(frame, input, cv::Size(1920, 1080));
            static auto inputBlob4 = infer_request4.GetBlob("src");
            preprocess(input, inputBlob4);
            infer_request4.StartAsync();
            flag4 = true;
        }

        if (reader->read(frame))
        {
            cv::resize(frame, input, cv::Size(1920, 1080));
            static auto inputBlob5 = infer_request5.GetBlob("src");
            preprocess(input, inputBlob5);
            infer_request5.StartAsync();
            flag5 = true;
        }

        if (reader->read(frame))
        {
            cv::resize(frame, input, cv::Size(1920, 1080));
            static auto inputBlob6 = infer_request6.GetBlob("src");
            preprocess(input, inputBlob6);
            infer_request6.StartAsync();
            flag6 = true;
        }

        if (reader->read(frame))
        {
            cv::resize(frame, input, cv::Size(1920, 1080));
            static auto inputBlob7 = infer_request7.GetBlob("src");
            preprocess(input, inputBlob7);
            infer_request7.StartAsync();
            flag7 = true;
        }

        infer_request.Wait();
        const InferenceEngine::Blob::Ptr dis_pred_blob = infer_request.GetBlob("pha");
        auto mdis_pred = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob);
        auto mdis_pred_holder = mdis_pred->rmap();
        const float* dis_pred = mdis_pred_holder.as<const float*>();
        int index = 0;
        unsigned char temp = 0;
        for (int h = 0; h < 1080; ++h) {
            for (int w = 0; w < 1920; ++w) {
                temp = (unsigned char)(dis_pred[index] * 255.0f);
                alpha.at<cv::Vec3b>(h, w) = { temp, temp, temp }; // R,G,B
                index++;
            }
        }
        cv::resize(alpha, output, cv::Size(input_width, input_height));
        writer->write(output);

        if (flag1)
        {
            infer_request1.Wait();
            const InferenceEngine::Blob::Ptr dis_pred_blob1 = infer_request1.GetBlob("pha");
            auto mdis_pred1 = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob1);
            auto mdis_pred_holder1 = mdis_pred1->rmap();
            const float* dis_pred1 = mdis_pred_holder1.as<const float*>();
            index = 0;
            temp = 0;
            for (int h = 0; h < 1080; ++h) {
                for (int w = 0; w < 1920; ++w) {
                    temp = (unsigned char)(dis_pred1[index] * 255.0f);
                    alpha.at<cv::Vec3b>(h, w) = { temp, temp, temp }; // R,G,B
                    index++;
                }
            }
            cv::resize(alpha, output, cv::Size(input_width, input_height));
            writer->write(output);
        }

        if (flag2)
        {
            infer_request2.Wait();
            const InferenceEngine::Blob::Ptr dis_pred_blob2 = infer_request2.GetBlob("pha");
            auto mdis_pred2 = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob2);
            auto mdis_pred_holder2 = mdis_pred2->rmap();
            const float* dis_pred2 = mdis_pred_holder2.as<const float*>();
            index = 0;
            temp = 0;
            for (int h = 0; h < 1080; ++h) {
                for (int w = 0; w < 1920; ++w) {
                    temp = (unsigned char)(dis_pred2[index] * 255.0f);
                    alpha.at<cv::Vec3b>(h, w) = { temp, temp, temp }; // R,G,B
                    index++;
                }
            }
            cv::resize(alpha, output, cv::Size(input_width, input_height));
            writer->write(output);
        }

        if (flag3)
        {
            infer_request3.Wait();
            const InferenceEngine::Blob::Ptr dis_pred_blob3 = infer_request3.GetBlob("pha");
            auto mdis_pred3 = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob3);
            auto mdis_pred_holder3 = mdis_pred3->rmap();
            const float* dis_pred3 = mdis_pred_holder3.as<const float*>();
            index = 0;
            temp = 0;
            for (int h = 0; h < 1080; ++h) {
                for (int w = 0; w < 1920; ++w) {
                    temp = (unsigned char)(dis_pred3[index] * 255.0f);
                    alpha.at<cv::Vec3b>(h, w) = { temp, temp, temp }; // R,G,B
                    index++;
                }
            }
            cv::resize(alpha, output, cv::Size(input_width, input_height));
            writer->write(output);
        }

        if (flag4)
        {
            infer_request4.Wait();
            const InferenceEngine::Blob::Ptr dis_pred_blob4 = infer_request4.GetBlob("pha");
            auto mdis_pred4 = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob4);
            auto mdis_pred_holder4 = mdis_pred4->rmap();
            const float* dis_pred4 = mdis_pred_holder4.as<const float*>();
            index = 0;
            temp = 0;
            for (int h = 0; h < 1080; ++h) {
                for (int w = 0; w < 1920; ++w) {
                    temp = (unsigned char)(dis_pred4[index] * 255.0f);
                    alpha.at<cv::Vec3b>(h, w) = { temp, temp, temp }; // R,G,B
                    index++;
                }
            }
            cv::resize(alpha, output, cv::Size(input_width, input_height));
            writer->write(output);
        }

        if (flag5)
        {
            infer_request5.Wait();
            const InferenceEngine::Blob::Ptr dis_pred_blob5 = infer_request5.GetBlob("pha");
            auto mdis_pred5 = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob5);
            auto mdis_pred_holder5 = mdis_pred5->rmap();
            const float* dis_pred5 = mdis_pred_holder5.as<const float*>();
            index = 0;
            temp = 0;
            for (int h = 0; h < 1080; ++h) {
                for (int w = 0; w < 1920; ++w) {
                    temp = (unsigned char)(dis_pred5[index] * 255.0f);
                    alpha.at<cv::Vec3b>(h, w) = { temp, temp, temp }; // R,G,B
                    index++;
                }
            }
            cv::resize(alpha, output, cv::Size(input_width, input_height));
            writer->write(output);
        }

        if (flag6)
        {
            infer_request6.Wait();
            const InferenceEngine::Blob::Ptr dis_pred_blob6 = infer_request6.GetBlob("pha");
            auto mdis_pred6 = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob6);
            auto mdis_pred_holder6 = mdis_pred6->rmap();
            const float* dis_pred6 = mdis_pred_holder6.as<const float*>();
            index = 0;
            temp = 0;
            for (int h = 0; h < 1080; ++h) {
                for (int w = 0; w < 1920; ++w) {
                    temp = (unsigned char)(dis_pred6[index] * 255.0f);
                    alpha.at<cv::Vec3b>(h, w) = { temp, temp, temp }; // R,G,B
                    index++;
                }
            }
            cv::resize(alpha, output, cv::Size(input_width, input_height));
            writer->write(output);
        }

        if (flag7)
        {
            infer_request7.Wait();
            const InferenceEngine::Blob::Ptr dis_pred_blob7 = infer_request7.GetBlob("pha");
            auto mdis_pred7 = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob7);
            auto mdis_pred_holder7 = mdis_pred7->rmap();
            const float* dis_pred7 = mdis_pred_holder7.as<const float*>();
            index = 0;
            temp = 0;
            for (int h = 0; h < 1080; ++h) {
                for (int w = 0; w < 1920; ++w) {
                    temp = (unsigned char)(dis_pred7[index] * 255.0f);
                    alpha.at<cv::Vec3b>(h, w) = { temp, temp, temp }; // R,G,B
                    index++;
                }
            }
            cv::resize(alpha, output, cv::Size(input_width, input_height));
            writer->write(output);
        }
    }
    reader->release();
    writer->release();
}
*/


