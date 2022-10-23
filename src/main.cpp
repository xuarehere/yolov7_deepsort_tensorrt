/*
 * @Author: xuarehere
 * @Date: 2022-09-18 04:14:53
 * @LastEditTime: 2022-10-23 12:57:29
 * @LastEditors: xuarehere
 * @Description: 
 * @FilePath: /yolov7_deepsort_tensorrt/src/main.cpp
 * 
 */
#include <iostream>
#include "yaml-cpp/yaml.h"
#include "video.h"

#include<stdio.h>
#include<string.h>
#include<unistd.h>

int main(int argc, char **argv) {
    if (argc < 3)
    {
        std::cout << "Please design config file and video name!" << std::endl;
        return -1;
    }
    std::string config_file = argv[1];
    std::string video_name = argv[2];
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node detect_config = root["detect"];
    YAML::Node tracker_config = root["tracker"];
    YAML::Node fastreid_config = root["fastreid"];
    YOLO detect(detect_config);
    detect.LoadEngine();
    ObjectTracker tracker(tracker_config);
    fastreid fastreid(fastreid_config);
    fastreid.LoadEngine();
    InferenceVideo(video_name, detect, tracker, fastreid);
    return  0;
}