/*
 * @Author: xuarehere
 * @Date: 2022-09-18 04:14:53
 * @LastEditTime: 2022-10-23 13:02:19
 * @LastEditors: xuarehere
 * @Description: 
 * @FilePath: /yolov7_deepsort_tensorrt/src/model.cpp
 * 
 */
//
// Created by linghu8812 on 2021/2/8.
//

#include "model.h"
#include "common.h"

void Model::onnxToTRTModel() {
    // create the builder
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    auto config = builder->createBuilderConfig();

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(gLogger.getReportableSeverity()))) {
        gLogError << "Failure while parsing ONNX file" << std::endl;
    }
    // Build the engine
    builder->setMaxBatchSize(BATCH_SIZE);
    config->setMaxWorkspaceSize(1_GiB);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    std::cout << "start building engine" << std::endl;
    engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build engine done" << std::endl;
    assert(engine);
    // we can destroy the parser
    parser->destroy();
    // save engine
    nvinfer1::IHostMemory *data = engine->serialize();
    std::ofstream file;
    file.open(engine_file, std::ios::binary | std::ios::out);
    std::cout << "writing engine file..." << std::endl;
    file.write((const char *) data->data(), data->size());
    std::cout << "save engine file done" << std::endl;
    file.close();
    // then close everything down
    network->destroy();
    builder->destroy();
}

bool Model::readTrtFile() {
    std::string cached_engine;
    std::fstream file;
    std::cout << "loading filename from:" << engine_file << std::endl;
    nvinfer1::IRuntime *trtRuntime;
    file.open(engine_file, std::ios::binary | std::ios::in);

    if (!file.is_open()) {
        std::cout << "read file error: " << engine_file << std::endl;
        cached_engine = "";
    }

    while (file.peek() != EOF) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        cached_engine.append(buffer.str());
    }
    file.close();

    trtRuntime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
    engine = trtRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
    std::cout << "deserialize done" << std::endl;

}

void Model::LoadEngine() {
    // create and load engine
    std::fstream existEngine;
    existEngine.open(engine_file, std::ios::in);
    if (existEngine) {
        readTrtFile();
        assert(engine != nullptr);
    } else {
        onnxToTRTModel();
        assert(engine != nullptr);
    }

    context = engine->createExecutionContext();
    assert(context != nullptr);

    //get buffers
    assert(engine->getNbBindings() == 2);
    int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);
    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * getElementSize(dtype);
        bufferSize[i] = totalSize;
        std::cout << "binding" << i << ": " << totalSize << std::endl;
        cudaMalloc(&buffers[i], totalSize);
    }
    //get stream
    cudaStreamCreate(&stream);
    outSize = int(bufferSize[1] / sizeof(float) / BATCH_SIZE);
}
