#include "inference_harness.h"

#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

RayBudgetInferenceEngine::RayBudgetInferenceEngine(const std::string& enginePath) {
    const std::vector<char> engineBytes = readBinaryFile(enginePath);

    runtime_.reset(nvinfer1::createInferRuntime(*this));
    if (!runtime_) {
        throw std::runtime_error("createInferRuntime returned nullptr");
    }

    engine_.reset(runtime_->deserializeCudaEngine(engineBytes.data(), engineBytes.size()));
    if (!engine_) {
        throw std::runtime_error("deserializeCudaEngine returned nullptr");
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("createExecutionContext returned nullptr");
    }

    const nvinfer1::Dims4 inputDims{kBatch, kInputChannels, kInputHeight, kInputWidth};
    if (!context_->setInputShape(kInputTensorName, inputDims)) {
        throw std::runtime_error("Failed to set input shape on execution context");
    }
    if (!context_->allInputDimensionsSpecified()) {
        throw std::runtime_error("Not all input dimensions are specified");
    }
    validateEngineContract();
    validateContextShapes();
}

RayBudgetInferenceEngine::~RayBudgetInferenceEngine() noexcept {
    releaseBuffers();
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

void RayBudgetInferenceEngine::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kINFO) {
        std::cerr << "[RayBudgetInferenceEngine] "
                  << severityToString(severity) << ": "
                  << (msg != nullptr ? msg : "<null>") << '\n';
    }
}

void RayBudgetInferenceEngine::allocateBuffers() {
    releaseBuffers();
    RAY_BUDGET_CHECK_CUDA(cudaMalloc(&d_input_, kInputBytes));
    RAY_BUDGET_CHECK_CUDA(cudaMalloc(&d_output_, kOutputBytes));
}

void RayBudgetInferenceEngine::releaseBuffers() noexcept {
    if (d_input_ != nullptr) {
        cudaError_t status = cudaFree(d_input_);
        if (status != cudaSuccess) {
            log(Severity::kERROR, cudaGetErrorString(status));
        }
        d_input_ = nullptr;
    }
    if (d_output_ != nullptr) {
        cudaError_t status = cudaFree(d_output_);
        if (status != cudaSuccess) {
            log(Severity::kERROR, cudaGetErrorString(status));
        }
        d_output_ = nullptr;
    }
}

void RayBudgetInferenceEngine::forwardAsync(
    void* d_inputG_Buffers,
    float* d_outputLogits,
    cudaStream_t stream) {
    if (d_inputG_Buffers == nullptr) {
        throw std::invalid_argument("forwardAsync received null input device pointer");
    }
    if (d_outputLogits == nullptr) {
        throw std::invalid_argument("forwardAsync received null output device pointer");
    }
    if (stream == nullptr) {
        throw std::invalid_argument("forwardAsync received null cudaStream_t");
    }

    bindAndEnqueue(d_inputG_Buffers, static_cast<void*>(d_outputLogits), stream);
}

void RayBudgetInferenceEngine::forwardAsync(cudaStream_t stream) {
    if (!hasOwnedBuffers()) {
        throw std::runtime_error("Owned buffers are not allocated. Call allocateBuffers() first.");
    }
    forwardAsync(d_input_, static_cast<float*>(d_output_), stream);
}

std::vector<char> RayBudgetInferenceEngine::readBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + path);
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("Engine file is empty: " + path);
    }

    std::vector<char> bytes(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(bytes.data(), size)) {
        throw std::runtime_error("Failed to read engine file: " + path);
    }
    return bytes;
}

void RayBudgetInferenceEngine::checkCuda(
    cudaError_t status,
    const char* expr,
    const char* file,
    int line) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA call failed: " << expr << " at " << file << ":" << line
            << " : " << cudaGetErrorString(status)
            << " (" << static_cast<int>(status) << ")";
        throw std::runtime_error(oss.str());
    }
}

const char* RayBudgetInferenceEngine::severityToString(Severity severity) noexcept {
    switch (severity) {
        case Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
        case Severity::kERROR: return "ERROR";
        case Severity::kWARNING: return "WARNING";
        case Severity::kINFO: return "INFO";
        case Severity::kVERBOSE: return "VERBOSE";
        default: return "UNKNOWN";
    }
}

void RayBudgetInferenceEngine::validateEngineContract() const {
    if (engine_->getNbIOTensors() != 2) {
        throw std::runtime_error("Expected exactly 2 IO tensors in TensorRT engine");
    }

    const nvinfer1::TensorIOMode inputMode = engine_->getTensorIOMode(kInputTensorName);
    const nvinfer1::TensorIOMode outputMode = engine_->getTensorIOMode(kOutputTensorName);
    if (inputMode != nvinfer1::TensorIOMode::kINPUT) {
        throw std::runtime_error("Engine does not contain required input tensor");
    }
    if (outputMode != nvinfer1::TensorIOMode::kOUTPUT) {
        throw std::runtime_error("Engine does not contain required output tensor");
    }

    if (engine_->getTensorDataType(kInputTensorName) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("Input tensor must be FP32");
    }
    if (engine_->getTensorDataType(kOutputTensorName) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("Output tensor must be FP32");
    }

    const nvinfer1::Dims inputDims = engine_->getTensorShape(kInputTensorName);
    const nvinfer1::Dims outputDims = engine_->getTensorShape(kOutputTensorName);
    if (inputDims.nbDims != 4 ||
        (inputDims.d[0] != kBatch && inputDims.d[0] != -1) ||
        inputDims.d[1] != kInputChannels ||
        inputDims.d[2] != kInputHeight ||
        inputDims.d[3] != kInputWidth) {
        std::ostringstream oss;
        oss << "Unexpected input tensor shape: nbDims=" << inputDims.nbDims;
        throw std::runtime_error(oss.str());
    }

    if (outputDims.nbDims != 4 ||
        (outputDims.d[0] != kBatch && outputDims.d[0] != -1) ||
        outputDims.d[1] != kOutputClasses ||
        outputDims.d[2] != kOutputTileX ||
        outputDims.d[3] != kOutputTileY) {
        std::ostringstream oss;
        oss << "Unexpected output tensor shape: nbDims=" << outputDims.nbDims;
        throw std::runtime_error(oss.str());
    }
}

void RayBudgetInferenceEngine::validateContextShapes() const {
    const nvinfer1::Dims inputDims = context_->getTensorShape(kInputTensorName);
    const nvinfer1::Dims outputDims = context_->getTensorShape(kOutputTensorName);

    if (inputDims.nbDims != 4 ||
        inputDims.d[0] != kBatch ||
        inputDims.d[1] != kInputChannels ||
        inputDims.d[2] != kInputHeight ||
        inputDims.d[3] != kInputWidth) {
        throw std::runtime_error("Execution context input shape does not match [1,7,1080,1920]");
    }
    if (outputDims.nbDims != 4 ||
        outputDims.d[0] != kBatch ||
        outputDims.d[1] != kOutputClasses ||
        outputDims.d[2] != kOutputTileX ||
        outputDims.d[3] != kOutputTileY) {
        throw std::runtime_error("Execution context output shape does not match [1,6,120,68]");
    }
}

void RayBudgetInferenceEngine::bindAndEnqueue(
    void* dInput,
    void* dOutput,
    cudaStream_t stream) {
    if (!context_->setTensorAddress(kInputTensorName, dInput)) {
        throw std::runtime_error("setTensorAddress failed for input_gbuffers");
    }
    if (!context_->setTensorAddress(kOutputTensorName, dOutput)) {
        throw std::runtime_error("setTensorAddress failed for output_tile_logits");
    }
    if (!context_->enqueueV3(stream)) {
        throw std::runtime_error("enqueueV3 failed");
    }
}
