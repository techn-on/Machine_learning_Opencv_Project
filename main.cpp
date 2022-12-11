#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;
using namespace dnn;

int main()
{
    VideoCapture cap("Project3_1.mp4");
    double fps = cap.get(CAP_PROP_FPS);
    int delay = 1000 / fps;
    float theta = 0;
    Mat frame, result, frame_clone2, edge_left, edge_mid, edge_right;
    Mat line_detect_left, line_detect_mid, line_detect_right;
    Mat inputBlob, detectionMat;
    Rect rect_left(145, 0, 145, 480);
    Rect rect_mid(290, 240, 145, 240);
    Rect rect_right(435, 0, 145, 480);
    vector<Vec2f> line_detected_left, line_detected_mid, line_detected_right;

    String labelname;

    String modelConfiguration = "YOLO/yolov2-tiny.cfg";
    String modelBinary = "YOLO/yolov2-tiny.weights";

    Net net = readNetFromDarknet(modelConfiguration, modelBinary);

    vector<String> classNamesVec;

    // class name load
    ifstream classNamesFile("YOLO/coco.names");
    if (classNamesFile.is_open())
    {
        string className = "";

        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }

    while (1)
    {

        cap >> frame;
        // frame.width = 720 , frame.height =480

        if (frame.empty())
            break;
        // frame clone
        result = frame.clone();
        frame_clone2 = frame.clone();

        if (frame_clone2.channels() == 4)
            cvtColor(frame_clone2, frame_clone2, COLOR_BGRA2BGR);
        // grayscale
        cvtColor(frame, frame, CV_BGR2GRAY);

        inputBlob = blobFromImage(frame_clone2, 1 / 255.F, Size(416, 416), Scalar(), true, false);
        // set the network input
        net.setInput(inputBlob, "data");
        // compute output
        detectionMat = net.forward("detection_out");
        // by default
        float confidenceThreshold = 0.24;
        for (int i = 0; i < detectionMat.rows; i++)
        {
            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
            size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;

            // prediction probability of each class
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

            // for drawing labels with name and confidence
            if (confidence > confidenceThreshold)
            {
                float x_center = detectionMat.at<float>(i, 0) * result.cols;
                float y_center = detectionMat.at<float>(i, 1) * result.rows;
                float width = detectionMat.at<float>(i, 2) * result.cols;
                float height = detectionMat.at<float>(i, 3) * result.rows;
                Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
                Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));
                Rect object(p1, p2);

                // green box detect
                Scalar object_roi_color(0, 255, 0);

                String className = objectClass < classNamesVec.size() ? classNamesVec[objectClass] : cv::format("unknown(%d)", objectClass);
                String label = format("%s: %.2f", className.c_str(), confidence);
                labelname = format("%s", className.c_str());

                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                // when car && person detect
                if ((labelname.compare("car") == 0) && (labelname.compare("person")==0))
                {
                    rectangle(result, object, object_roi_color);
                    rectangle(result, Rect(p1, Size(labelSize.width, labelSize.height + baseLine)), object_roi_color, FILLED);
                    putText(result, label, p1 + Point(0, labelSize.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
                }
            } // if (confidence > confidenceThreshold)
        }     // for (int i = 0; i < detectionMat.rows; i++)

        line_detect_left = frame(rect_left);
        line_detect_mid = frame(rect_mid);
        line_detect_right = frame(rect_right);

        Canny(line_detect_left, edge_left, 90, 255, 3);   // edge 생성
        Canny(line_detect_mid, edge_mid, 90, 255, 3);     // edge 생성
        Canny(line_detect_right, edge_right, 90, 255, 3); // edge 생성
        imshow("line_detect_left", edge_left);
        imshow("line_detect_mid", edge_mid);
        imshow("line_detect_right", edge_right);

        HoughLines(edge_left, line_detected_left, 1, CV_PI / 180, 90, 0, 0, CV_PI / 180 * 30, CV_PI / 180 * 60);       // 직선들 생성
        HoughLines(edge_mid, line_detected_mid, 1, CV_PI / 180, 90, 0, 0, CV_PI / 180 * (-15), CV_PI / 180 * (15));    // 직선들
        HoughLines(edge_right, line_detected_right, 1, CV_PI / 180, 90, 0, 0, CV_PI / 180 * (120), CV_PI / 180 * 150); // 직선들 생성
        if ((line_detected_left.size() == 0) && (line_detected_mid.size() > 0) && (line_detected_right.size() == 0))
        {
            putText(result, format("\"Warning!: Lane departure\""), Point(100, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 4);
        }
        cout << line_detected_left.size() << endl;
        cout << line_detected_mid.size() << endl;
        cout << line_detected_right.size() << endl;

        imshow("", result);

        if (waitKey(1) >= 0)
            break;
    }
    return 0;
}
