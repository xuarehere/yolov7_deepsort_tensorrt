#!/bin/bash

###
 # @Author: xuarehere
 # @Date: 2022-10-23 14:45:29
 # @LastEditTime: 2022-10-23 16:55:46
 # @LastEditors: xuarehere
 # @Description: 
 # @FilePath: /yolov7_deepsort_tensorrt/scripts/build.sh
 # 
### 
dir_path="../build"
file_name="$0"
dir_status=-1
input_example="
\t 1) bash $file_name \n
\t It is to build the directory \"../build\". \n
\n
\t 2) bash $file_name rm \n
\t The directory \"../build\" exists. And it will remove the directory \"../build\" and build it again.
"

function build_again() {
    cd $1 && make -j$(nproc) && cd -
}


function build_new () {
    cd $1 && cmake .. && make -j$(nproc) && cd -
}

function build(){
    if [ $1 == "0" ]
    then
        build_new $2
    elif [ $1 == "1" ]
    then
        build_again $2
    else
      echo 'no build options'
    fi
}

function check_dir_exist () {
    # echo "$1" # arguments are accessible through $1, $2,...
    if [ -d $1 ];then
        echo  "$1 exists"
        echo "1"
        return $?
    else
        echo "$1 does not exist"
        mkdir $1
        echo "0"
        return $?
    fi    
}

function main(){
    args_nums=($#)
    args="$@"
    echo "num of args: $args_nums"

    dir_status=$(check_dir_exist $dir_path)
    dir_status=$(echo ${dir_status: -1})  # get the last one str

    if [ $args_nums  -ge "2" ] # greater or equal
    then
        echo -e "\033[31m Input args error! Please input again! \033[0m"
        echo -e $input_example
    elif [[ $args_nums -eq "1" && $args = "rm" ]]   # rm dir and rebuild 
    then
        rm -r $dir_path
        dir_status=$(check_dir_exist $dir_path)
        dir_status=$(echo ${dir_status: -1})  # get the last one str        
        build $dir_status $dir_path
    elif [ $args_nums  == "0" ]     # build
    then
        build $dir_status $dir_path
    else
        echo -e "\033[31m Input args error! Please input again! \033[0m"
        echo -e $input_example
    fi
}


main $@
