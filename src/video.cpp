/*
 * @Author: xuarehere
 * @Date: 2022-09-18 04:14:53
 * @LastEditTime: 2022-10-23 13:02:33
 * @LastEditors: xuarehere
 * @Description: 
 * @FilePath: /yolov7_deepsort_tensorrt/src/video.cpp
 * 
 */

#include "video.h"
#include <opencv2/opencv.hpp>

void InferenceVideo(const std::string &video_name, YOLO &yolo, ObjectTracker &tracker, fastreid &fastreid) {
    std::cout << "Processing: " << video_name << std::endl;
    cv::VideoCapture video_cap(video_name);
    cv::Size sSize = cv::Size((int) video_cap.get(cv::CAP_PROP_FRAME_WIDTH),
                              (int) video_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Frame width is: " << sSize.width << ", height is: " << sSize.height << std::endl;
    auto fFps = (float)video_cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter video_writer("result.avi",  cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            fFps, sSize);
    cv::Mat src_img;
    auto frame_i=0;
    while (video_cap.read(src_img)) {
        std::cout << "----------frame_i: " << frame_i << "----------" << std::endl;
        cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
        std::vector<cv::Mat> vec_org_img;
        vec_org_img.push_back(src_img);
        auto detect_boxes = yolo.InferenceImages(vec_org_img);        // BOX 过大
        auto object_features = fastreid.InferenceImages(vec_org_img, detect_boxes);
        tracker.update(detect_boxes[0], object_features[0], vec_org_img[0].cols, vec_org_img[0].rows);
        tracker.DrawResults(vec_org_img[0]);
        cv::imwrite("../results.jpg", vec_org_img[0]);
        video_writer.write(vec_org_img[0]);
        frame_i+=1;
        std::cout << "" << std::endl;
    }
}