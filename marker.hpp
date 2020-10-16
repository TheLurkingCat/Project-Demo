#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "darknet.hpp"

void mark(cv::Mat image,
          std::vector<bbox_t> bounding_boxes,
          const std::vector<std::string> &info,
          const std::vector<std::string> &MAC) {
    const cv::Scalar box_color = CV_RGB(255, 0, 255);
    const cv::Scalar text_color = CV_RGB(255, 0, 0);

    constexpr double scale = 1.0;
    constexpr int thick = 3;

    std::size_t shortest = std::min(bounding_boxes.size(), MAC.size());
    for (std::size_t i = 0; i < shortest; ++i) {
        const bbox_t &box = bounding_boxes[i];
        int mac_baseline, text_baseline;
        cv::Size mac_box = cv::getTextSize(MAC[i], cv::FONT_HERSHEY_SIMPLEX, scale, thick, &mac_baseline);
        cv::Point mac_position(box.x - (mac_box.width - box.w) / 2, box.y - mac_baseline);
        cv::putText(image, MAC[i], mac_position, cv::FONT_HERSHEY_SIMPLEX, scale, text_color, thick, cv::LINE_AA);

        cv::Size text_box = cv::getTextSize(info[i], cv::FONT_HERSHEY_SIMPLEX, scale, thick, &text_baseline);
        cv::Point text_position(box.x - (text_box.width - box.w) / 2,
                                box.y - text_baseline - mac_baseline - mac_box.height);
        cv::putText(image, info[i], text_position, cv::FONT_HERSHEY_SIMPLEX, scale, text_color, thick, cv::LINE_AA);

        cv::rectangle(image, cv::Rect(box.x, box.y, box.w, box.h), box_color, 2, cv::LINE_8);
    }
}
