###
 # @Author: xuarehere
 # @Date: 2022-10-21 12:17:18
 # @LastEditTime: 2022-10-23 09:00:05
 # @LastEditors: xuarehere
 # @Description: 
 # @FilePath: /yolov7_deepsort_tensorrt/scripts/build_again.sh
 # 可以输入预定的版权声明、个性签名、空行等
### 

cd ../build/ && make -j$(nproc) && cd -