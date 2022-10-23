#!/usr/bin/env bash
###
 # @Author: xuarehere
 # @Date: 2022-10-23 07:57:47
 # @LastEditTime: 2022-10-23 09:00:16
 # @LastEditors: xuarehere
 # @Description: 
 # @FilePath: /yolov7_deepsort_tensorrt/scripts/build_new.sh
 # 可以输入预定的版权声明、个性签名、空行等
### 
cd ../build/ && rm -r * && cmake .. && make -j$(nproc) && cd -
