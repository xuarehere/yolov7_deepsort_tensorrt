#!/bin/bash
###
 # @Author: xuarehere
 # @Date: 2022-10-24 04:59:04
 # @LastEditTime: 2022-10-24 04:59:11
 # @LastEditors: xuarehere
 # @Description: 
 # @FilePath: /yolov7_deepsort_tensorrt/scripts/get_weight.sh
 # 
### 

# https://github.com/xuarehere/yolov7_deepsort_tensorrt/releases/download/v0.0.1/fast-reid_mobilenetv2.onnx
# https://github.com/xuarehere/yolov7_deepsort_tensorrt/releases/download/v0.0.1/yolov5s.onnx
# https://github.com/xuarehere/yolov7_deepsort_tensorrt/releases/download/v0.0.1/yolov7.onnx
# Download/unzip images
d_save='../weights/' 
url=https://github.com/xuarehere/yolov7_deepsort_tensorrt/releases/download/v0.0.1/
f1='fast-reid_mobilenetv2.onnx'
f2='yolov7.onnx'  
f3='yolov5s.onnx'
for f in $f1 $f2 $f3; do
  save_path=$d_save$f 
  echo 'Downloading' $url$f '...'
  curl -L $url$f -o $f  && mv $f $save_path 
  echo "save:" $save_path
done
wait # finish background tasks
