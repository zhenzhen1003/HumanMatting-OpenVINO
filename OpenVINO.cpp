#include "OpenVINO.h"
#include <windows.h>

using namespace std;

const int N = 2;

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
    //ie.SetConfig({ { CONFIG_KEY(GPU_THROUGHPUT_STREAMS),std::to_string(N) } }, "GPU");
    //ie.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS),std::to_string(N) } }, "CPU");

    executable_network = ie.LoadNetwork(model, device_name);
    infer_request = executable_network.CreateInferRequest();
}

void OpenVINO::setVideoCapture(cv::VideoCapture *reader)
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

/*
void OpenVINO::predict()
{
    while (reader->read(curr_frame))
    {
        static auto inputBlob = infer_request.GetBlob("src");
        preprocess(curr_frame, inputBlob);

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
        writer->write(alpha);
    }
    reader->release();
    writer->release();
}
*/


void OpenVINO::predict()
{
    cv::Mat output;
    cv::Mat input;
    while (reader->read(frame))
    {
        cv::resize(frame, input, cv::Size(1920, 1080));
        static auto inputBlob = infer_request.GetBlob("src");
        preprocess(input, inputBlob);
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
        cv::resize(alpha, output, cv::Size(input_width, input_height));
        writer->write(output);
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
    int thread_num = N;
    int k = 0;
    int i = 0;
    while (k < frame_num)
    {
        int cnt = min(thread_num, frame_num - k);
        for (i = 0; i < cnt; i++)
        {
            reader->read(curr_frame);
            static auto inputBlob = infer_request.at(i).GetBlob("src");
            preprocess(curr_frame, inputBlob);
            infer_request.at(i).StartAsync();
        }
        for (int j = 0; j < i; j++)
        {
            if (InferenceEngine::StatusCode::OK == infer_request.at(j).Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY))
            {
                const InferenceEngine::Blob::Ptr dis_pred_blob = infer_request.at(j).GetBlob("pha");
                auto mdis_pred = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob);
                auto mdis_pred_holder = mdis_pred->rmap();
                const float* dis_pred = mdis_pred_holder.as<const float*>();

                cout << dis_pred << endl;

                int index = 0;
                unsigned char temp = 0;
                for (int h = 0; h < 1080; ++h) {
                    for (int w = 0; w < 1920; ++w) {
                        temp = (unsigned char)(dis_pred[index] * 255.0f);
                        alpha.at<cv::Vec3b>(h, w) = { temp, temp, temp }; // R,G,B
                        index++;
                    }
                }
            }
            cv::imshow("frame", alpha);
            cv::waitKey(1);
        }
        k = k + i;
    }
}
*/



/*
void OpenVINO::predict()
{
    while(reader->read(curr_frame))
    {
        DWORD start_times = GetTickCount();
        static auto inputBlob = infer_request.GetBlob("src");
        preprocess(curr_frame, inputBlob);
        infer_request.Infer();
        const InferenceEngine::Blob::Ptr dis_pred_blob = infer_request.GetBlob("pha");
        auto mdis_pred = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob);
        auto mdis_pred_holder = mdis_pred->rmap();
        const float* dis_pred = mdis_pred_holder.as<const float*>();

        int index = 0;
        for (int h = 0; h < 1080; ++h) {
            for (int w = 0; w < 1920; ++w) {
                alpha.at<cv::Vec3f>(h, w) = {dis_pred[index], dis_pred[index], dis_pred[index]}; // R,G,B
                index++; // update STEP times
            }
        }
        DWORD end_times = GetTickCount();
        cout << "The run time is:" << end_times - start_times << "ms!" << endl;
    }
}
*/

/*
void OpenVINO::predict()
{
    QTime time;
    while(reader->read(curr_frame))
    {
        time.start();
        static auto inputBlob = infer_request.GetBlob("src");
        preprocess(curr_frame, inputBlob);

        infer_request.Infer();


        const InferenceEngine::Blob::Ptr dis_pred_blob = infer_request.GetBlob("pha");
        auto mdis_pred = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob);
        auto mdis_pred_holder = mdis_pred->rmap();
        const float* dis_pred = mdis_pred_holder.as<const float*>();
        int index = 0;
        for (int h = 0; h < 1080; ++h) {
            for (int w = 0; w < 1920; ++w) {
                alpha.at<cv::Vec3f>(h, w) = {dis_pred[index], dis_pred[index], dis_pred[index]}; // R,G,B
                index++; // update STEP times
            }
        }
        qDebug()<<time.elapsed()/1000.0<<"s";
    }


    //cv::imshow("frame", alpha);
}
*/


/*
void OpenVINO::predict()
{
    for(int j = 0; j < 16; j++)
    {
    QTime time;
    time.start();
        //创建任务类对象
        for(int i = 0; i < 1; i++)
        {
            reader->read(frame[i]);
            thread_task.append(new Infer(i));
            QThreadPool::globalInstance()->start(thread_task.at(i));
        }
        while(QThreadPool::globalInstance()->activeThreadCount() != 0)
        {
            //qDebug() << QThreadPool::globalInstance()->activeThreadCount();
        }
    qDebug()<<time.elapsed()/1000.0<<"s";
    }

    //while(QThreadPool::globalInstance()->activeThreadCount() != 1)
    //{
        //qDebug() << QThreadPool::globalInstance()->activeThreadCount();
    //}
    qDebug() << 222;
}
*/


/*
void OpenVINO::predict()
{
    reader->read(curr_frame);
    cv::imshow("frame", curr_frame);
    static auto inputBlob = infer_request_curr->GetBlob(input_names[0]);
    preprocess(curr_frame, inputBlob);
    infer_request_curr->StartAsync();

    while(reader->read(curr_frame))
    {
        qDebug() << 11;
        static auto inputBlob = infer_request_next->GetBlob(input_names[0]);
        preprocess(curr_frame, inputBlob);
        infer_request_next->StartAsync();
        if(infer_request_curr->Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY))
        {
            const InferenceEngine::Blob::Ptr dis_pred_blob = infer_request_curr->GetBlob("pha");
            auto mdis_pred = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob);
            auto mdis_pred_holder = mdis_pred->rmap();
            const float* dis_pred = mdis_pred_holder.as<const float*>();

            //QTime time;
            //time.start();
            // reshape
            int index = 0;
            for (int h = 0; h < 1080; ++h) {
                for (int w = 0; w < 1920; ++w) {
                    alpha.at<cv::Vec3f>(h, w) = {dis_pred[index], dis_pred[index], dis_pred[index]}; // R,G,B
                    index++; // update STEP times
                }
            }
        }
        infer_request_curr.swap(infer_request_next);
    }
    cv::imshow("frame", alpha);
}
*/



/*
void OpenVINO::predict()
{
    int thread_num = 4;
    int k = 0;
    int i = 0;
    while(k < frame_num)
    {
        int cnt = min(thread_num,frame_num - k);
        for(i = 0; i < cnt && reader->read(curr_frame); i++)
        {
            //time.start();
            static auto inputBlob = asyn_request.at(i)->GetBlob(input_names[0]);
            preprocess(curr_frame, inputBlob);
            asyn_request.at(i)->StartAsync();
            //qDebug()<<time.elapsed()/1000.0<<"s";
        }
        for(int j = 0; j < i; j++)
        {
            //time.start();
            if(asyn_request.at(j)->Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY))
            {
                const InferenceEngine::Blob::Ptr dis_pred_blob = asyn_request.at(j)->GetBlob("pha");
                auto mdis_pred = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob);
                auto mdis_pred_holder = mdis_pred->rmap();
                const float* dis_pred = mdis_pred_holder.as<const float*>();

                int index = 0;
                for (int h = 0; h < 1080; ++h) {
                    for (int w = 0; w < 1920; ++w) {
                        alpha.at<cv::Vec3f>(h, w) = {dis_pred[index], dis_pred[index], dis_pred[index]}; // R,G,B
                        index++; // update STEP times
                    }
                }
            }
        }
        k = k + i;
    }
}
*/




/*
void OpenVINO::predict()
{
    QTime time;

    for(int i = 0; i < 30; i++)
    {
        reader->read(curr_frame);
        //time.start();
        InferenceEngine::InferRequest::Ptr temp = executable_network.CreateInferRequestPtr();
        //time.start();
        static auto inputBlob = temp->GetBlob(input_names[0]);
        preprocess(curr_frame, inputBlob);
        temp->StartAsync();
        asyn_request.enqueue(temp);
        //qDebug()<<time.elapsed()/1000.0<<"s";

    }
    int j = 29;
    while(!asyn_request.empty())
    {
        time.start();
        qDebug() << 11;
        InferenceEngine::InferRequest::Ptr curr = asyn_request.dequeue();
        if(curr->Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY))
        {
            const InferenceEngine::Blob::Ptr dis_pred_blob = curr->GetBlob("pha");
            auto mdis_pred = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob);
            auto mdis_pred_holder = mdis_pred->rmap();
            const float* dis_pred = mdis_pred_holder.as<const float*>();

            int index = 0;
            for (int h = 0; h < 1080; ++h) {
                for (int w = 0; w < 1920; ++w) {
                    alpha.at<cv::Vec3f>(h, w) = {dis_pred[index], dis_pred[index], dis_pred[index]}; // R,G,B
                    index++; // update STEP times
                }
            }
        }
        if(j < frame_num)
        {
            reader->read(curr_frame);
            //time.start();
            static auto inputBlob = curr->GetBlob(input_names[0]);
            preprocess(curr_frame, inputBlob);
            //qDebug()<<time.elapsed()/1000.0<<"s";
            curr->StartAsync();
            asyn_request.enqueue(curr);
            j++;
        }
    }
    qDebug()<<time.elapsed()/1000.0<<"s";
}
*/


/*
void OpenVINO::predict()
{
    reader->read(curr_frame);
    static auto inputBlob = infer_request_curr->GetBlob(input_names[0]);
    preprocess(curr_frame, inputBlob);

    infer_request_curr->StartAsync();

    while(reader->read(curr_frame))
    {
        qDebug() << 11;
        static auto inputBlob = infer_request_next->GetBlob(input_names[0]);
        preprocess(curr_frame, inputBlob);
        infer_request_next->StartAsync();
        if(infer_request_curr->Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY))
        {
            const InferenceEngine::Blob::Ptr dis_pred_blob = infer_request_curr->GetBlob("pha");
            auto mdis_pred = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob);
            auto mdis_pred_holder = mdis_pred->rmap();
            const float* dis_pred = mdis_pred_holder.as<const float*>();

            //QTime time;
            //time.start();
            // reshape
            int index = 0;
            for (int h = 0; h < 1080; ++h) {
                for (int w = 0; w < 1920; ++w) {
                    alpha.at<cv::Vec3f>(h, w) = {dis_pred[index], dis_pred[index], dis_pred[index]}; // R,G,B
                    index++; // update STEP times
                }
            }

            //Find number of pixels.
            int numberOfPixels = curr_frame.rows * curr_frame.cols * curr_frame.channels();

            //Get floating point pointers to the data matrices
            uint8_t* fptr = reinterpret_cast<uint8_t*>(curr_frame.data);
            float* aptr = reinterpret_cast<float*>(alpha.data);
            uint8_t* outImagePtr = reinterpret_cast<uint8_t*>(out.data);

            //Loop over all pixesl ONCE
            for(
              int i = 0;
              i < numberOfPixels;
              i++, outImagePtr++, fptr++, aptr++
            )
            {
                *outImagePtr = (*fptr)*(*aptr) + (255)*(1 - *aptr);
            }
        }
        infer_request_curr.swap(infer_request_next);
    }
    */





    //qDebug()<<time.elapsed()/1000.0<<"s";

    //cv::imshow("frame", out);


    /*
    static auto inputBlob = infer_request_curr->GetBlob(input_names[0]);
    preprocess(image, inputBlob);

    infer_request_curr->Infer();


    const InferenceEngine::Blob::Ptr dis_pred_blob = infer_request_curr->GetBlob("pha");
    auto mdis_pred = InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob);
    auto mdis_pred_holder = mdis_pred->rmap();
    const float* dis_pred = mdis_pred_holder.as<const float*>();

    //QTime time;
    //time.start();
    // reshape
    int index = 0;
    for (int h = 0; h < 1080; ++h) {
        for (int w = 0; w < 1920; ++w) {
            alpha.at<cv::Vec3f>(h, w) = {dis_pred[index], dis_pred[index], dis_pred[index]}; // R,G,B
            index++; // update STEP times
        }
    }

    //Find number of pixels.
    int numberOfPixels = image.rows * image.cols * image.channels();

    //Get floating point pointers to the data matrices
    uint8_t* fptr = reinterpret_cast<uint8_t*>(image.data);
    float* aptr = reinterpret_cast<float*>(alpha.data);
    uint8_t* outImagePtr = reinterpret_cast<uint8_t*>(out.data);

    //Loop over all pixesl ONCE
    for(
      int i = 0;
      i < numberOfPixels;
      i++, outImagePtr++, fptr++, aptr++
    )
    {
        *outImagePtr = (*fptr)*(*aptr) + (255)*(1 - *aptr);
    }

    //qDebug()<<time.elapsed()/1000.0<<"s";

    //cv::imshow("frame", out);
    */
//}
