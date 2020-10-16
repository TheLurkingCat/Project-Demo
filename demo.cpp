#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "darknet.hpp"
#include "marker.hpp"

int main(int argc, char *argv[]) {
    Detector detector("../resource/YOLO/model.cfg", "../resource/YOLO/model.weights");
    std::ifstream user("../resource/WiFi/user.txt");
    int frame_no, rssi[3], mcs[3];
    while (true) {
        user >> frame_no >> rssi[0] >> mcs[0] >> frame_no >> rssi[1] >> mcs[1] >> frame_no >> rssi[2] >> mcs[2];
        cv::Mat frame = cv::imread("../resource/media/" + std::to_string(frame_no) + ".jpg");
        if (frame.empty())
            break;
        std::vector<bbox_t> prediction = detector.detect(frame);
        std::sort(prediction.begin(), prediction.end(), [](const bbox_t &lhs, const bbox_t &rhs) {
            return lhs.w + lhs.x > rhs.w + rhs.x;
        });

        mark(frame,
             prediction,
             {"RSSI: " + std::to_string(rssi[0]) + " MCS: " + std::to_string(mcs[0]),
              "RSSI: " + std::to_string(rssi[1]) + " MCS: " + std::to_string(mcs[1]),
              "RSSI: " + std::to_string(rssi[2]) + " MCS: " + std::to_string(mcs[2])});

        // cv::imwrite("../out/" + std::to_string(frame_no) + ".jpg", frame, {cv::IMWRITE_JPEG_QUALITY, 100});
        cv::imshow("Demo", frame);
        cv::waitKey(0);
        break;
    }
    return 0;
}