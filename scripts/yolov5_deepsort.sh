
###
 # @Author: xuarehere
 # @Date: 2022-10-23 07:50:03
 # @LastEditTime: 2022-10-23 09:00:25
 # @LastEditors: xuarehere
 # @Description: 
 # @FilePath: /yolov7_deepsort_tensorrt/scripts/yolov5_deepsort.sh
 # 可以输入预定的版权声明、个性签名、空行等
### 
cd ../build 
./deepsort ../configs/config_v5.yaml  ../001.avi
cd -