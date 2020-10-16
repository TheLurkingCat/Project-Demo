#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "darknet.hpp"

void mark(cv::Mat image, std::vector<bbox_t> bounding_boxes, const std::vector<std::string> &MAC) {
    const cv::Scalar box_color = CV_RGB(255, 0, 255);
    const cv::Scalar text_color = CV_RGB(255, 0, 0);

    std::size_t shortest = std::min(bounding_boxes.size(), MAC.size());
    for (std::size_t i = 0; i < shortest; ++i) {
        const bbox_t &box = bounding_boxes[i];
        int baseline;
        cv::Size text_box = cv::getTextSize(MAC[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::Point text_position(box.x - (text_box.width - box.w) / 2, box.y - baseline);
        cv::putText(image, MAC[i], text_position, cv::FONT_HERSHEY_SIMPLEX, 1.1, text_color, 3, cv::LINE_AA);
        cv::rectangle(image, cv::Rect(box.x, box.y, box.w, box.h), box_color, 2, cv::LINE_8);
    }
}