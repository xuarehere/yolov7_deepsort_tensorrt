//
// Created by linghu8812 on 2021/10/29.
//

#ifndef OBJECT_TRACKER_VIDEO_H
#define OBJECT_TRACKER_VIDEO_H

#include "yolo.h"
#include "tracker.h"
#include "fast-reid.h"

void InferenceVideo(const std::string &video_name, YOLO &yolov5, ObjectTracker &tracker, fastreid &fastreid);

#endif //OBJECT_TRACKER_VIDEO_H
