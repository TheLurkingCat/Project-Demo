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
    while (user >> frame_no >> rssi[0] >> mcs[0] >> frame_no >> rssi[1] >> mcs[1] >> frame_no >> rssi[2] >> mcs[2]) {
        frame_no *= 5;
        for (int i = frame_no; i < frame_no + 5; ++i) {
            cv::Mat frame = cv::imread("../resource/media/" + std::to_string(i) + ".png");
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
                  "RSSI: " + std::to_string(rssi[2]) + " MCS: " + std::to_string(mcs[2])},
                 {"MAC: 38:78:62:46:3d:cc", "MAC: b0:6e:bf:04:bc:4c", "MAC: 04:92:26:7e:18:00"});

            cv::imwrite("../out/" + std::to_string(i) + ".png", frame, {cv::IMWRITE_PNG_COMPRESSION, 9});
            cv::imshow("Demo", frame);
            cv::waitKey(1);
        }
    }
    return 0;
}
