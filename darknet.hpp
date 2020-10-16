/*
    This file is a modification of yolo_v2_class.hpp
*/
#pragma once

#include <deque>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define C_SHARP_MAX_OBJECTS 1000

struct bbox_t {
    unsigned int x, y, w, h;        // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                     // confidence - probability that the object was found correctly
    unsigned int obj_id;            // class of object - from range [0, classes-1]
    unsigned int track_id;          // tracking id for video (0 - untracked, 1 - inf - tracked object)
    unsigned int frames_counter;    // counter of frames on which the object was detected
    float x_3d, y_3d, z_3d;         // center of object (in Meters) if ZED 3D Camera is used
};

struct image_t {
    int h;          // height
    int w;          // width
    int c;          // number of chanels (3 - for RGB)
    float *data;    // pointer to the image data
};

struct bbox_t_container {
    bbox_t candidates[C_SHARP_MAX_OBJECTS];
};

extern "C" {
int init(const char *configurationFilename, const char *weightsFilename, int gpu);
int detect_image(const char *filename, bbox_t_container &container);
int detect_mat(const uint8_t *data, const size_t data_length, bbox_t_container &container);
int dispose();
int get_device_count();
int get_device_name(int gpu, char *deviceName);
bool built_with_cuda();
bool built_with_cudnn();
bool built_with_opencv();
void send_json_custom(char const *send_buf, int port, int timeout);
}

class Detector {
  public:
    // Imported from darknet.dll
    Detector(std::string cfg_filename, std::string weight_filename, int gpu_id = 0);
    ~Detector();

    std::vector<bbox_t> detect(std::string image_filename, float thresh = 0.2, bool use_mean = false);
    std::vector<bbox_t> detect(image_t img, float thresh = 0.25f, bool use_mean = false);
    static image_t load_image(std::string image_filename);
    static void free_image(image_t m);
    int get_net_width() const;
    int get_net_height() const;
    int get_net_color_depth() const;
    std::vector<bbox_t> tracking_id(std::vector<bbox_t> cur_bbox_vec,
                                    bool const change_history = true,
                                    int const frames_story = 5,
                                    int const max_dist = 40);
    void *get_cuda_context();

    // -------------------------------------------------------------------------

    std::vector<bbox_t>
    detect_resized(image_t img, int init_w, int init_h, float thresh = 0.25f, bool use_mean = false) {
        if (img.data == nullptr)
            throw std::runtime_error("Image is empty");
        auto detection_boxes = detect(img, thresh, use_mean);
        float wk = static_cast<float>(init_w) / img.w;
        float hk = static_cast<float>(init_h) / img.h;
        for (auto &&i : detection_boxes)
            i.x *= wk, i.w *= wk, i.y *= hk, i.h *= hk;
        return detection_boxes;
    }

    std::vector<bbox_t> detect(cv::Mat mat, float thresh = 0.25f, bool use_mean = false) {
        if (mat.data == nullptr)
            throw std::runtime_error("Image is empty");
        auto image_ptr = mat_to_image_resize(mat);
        return detect_resized(*image_ptr, mat.cols, mat.rows, thresh, use_mean);
    }

    std::shared_ptr<image_t> mat_to_image_resize(cv::Mat mat) const {
        if (mat.data == nullptr)
            return nullptr;

        cv::Size network_size = cv::Size(get_net_width(), get_net_height());
        cv::Mat det_mat;
        if (mat.size() != network_size)
            cv::resize(mat, det_mat, network_size);
        else
            det_mat = mat;    // only reference is copied

        return mat_to_image(det_mat);
    }

    static std::shared_ptr<image_t> mat_to_image(cv::Mat img_src) {
        cv::Mat img;
        if (img_src.channels() == 4)
            cv::cvtColor(img_src, img, cv::COLOR_RGBA2BGR);
        else if (img_src.channels() == 3)
            cv::cvtColor(img_src, img, cv::COLOR_RGB2BGR);
        else if (img_src.channels() == 1)
            cv::cvtColor(img_src, img, cv::COLOR_GRAY2BGR);
        else
            std::cerr << " Warning: img_src.channels() is not 1, 3 or 4. It is = " << img_src.channels() << std::endl;
        std::shared_ptr<image_t> image_ptr(new image_t, [](image_t *img) {
            free_image(*img);
            delete img;
        });
        *image_ptr = mat_to_image_custom(img);
        return image_ptr;
    }

    // Imported from darknet.dll
    const int cur_gpu_id;
    bool wait_stream;
    float nms = 0.4f;
    // -------------------------------------------------------------------------

  private:
    static image_t mat_to_image_custom(cv::Mat mat) {
        int w = mat.cols;
        int h = mat.rows;
        int pixels = w * h;
        int c = mat.channels();
        image_t im = make_image_custom(w, h, c);
        int step = static_cast<int>(mat.step);
        for (int y = 0, yw = 0, ystep = 0; y < h; ++y, yw += w, ystep += step)
            for (int k = 0, kp = 0; k < c; ++k, kp += pixels)
                for (int x = 0, xc = 0; x < w; ++x, xc += c)
                    im.data[kp + yw + x] = mat.data[ystep + xc + k] / 255.0f;
        return im;
    }

    static image_t make_image_custom(int w, int h, int c) {
        image_t out = {h, w, c, nullptr};
        out.data = static_cast<float *>(calloc(h * w * c, sizeof(float)));
        return out;
    }
    // Imported from darknet.dll
    std::shared_ptr<void> detector_gpu_ptr;
    std::deque<std::vector<bbox_t>> prev_bbox_vec_deque;
    std::string _cfg_filename, _weight_filename;
    // -------------------------------------------------------------------------
};
