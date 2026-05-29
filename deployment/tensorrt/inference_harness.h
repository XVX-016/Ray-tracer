#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class RayBudgetInferenceEngine final : public nvinfer1::ILogger {
public:
    static constexpr const char* kInputTensorName = "input_gbuffers";
    static constexpr const char* kOutputTensorName = "output_tile_logits";

    static constexpr int32_t kBatch = 1;
    static constexpr int32_t kInputChannels = 7;
    static constexpr int32_t kInputHeight = 1080;
    static constexpr int32_t kInputWidth = 1920;
    static constexpr int32_t kOutputClasses = 6;
    static constexpr int32_t kOutputTileX = 120;
    static constexpr int32_t kOutputTileY = 68;

    static constexpr size_t kInputElementCount =
        static_cast<size_t>(kBatch) * kInputChannels * kInputHeight * kInputWidth;
    static constexpr size_t kOutputElementCount =
        static_cast<size_t>(kBatch) * kOutputClasses * kOutputTileX * kOutputTileY;
    static constexpr size_t kInputBytes = kInputElementCount * sizeof(float);
    static constexpr size_t kOutputBytes = kOutputElementCount * sizeof(float);

    explicit RayBudgetInferenceEngine(const std::string& enginePath);
    ~RayBudgetInferenceEngine() noexcept override;

    RayBudgetInferenceEngine(const RayBudgetInferenceEngine&) = delete;
    RayBudgetInferenceEngine& operator=(const RayBudgetInferenceEngine&) = delete;
    RayBudgetInferenceEngine(RayBudgetInferenceEngine&&) = delete;
    RayBudgetInferenceEngine& operator=(RayBudgetInferenceEngine&&) = delete;

    void log(Severity severity, const char* msg) noexcept override;

    static void checkCuda(cudaError_t status, const char* expr, const char* file, int line);

    void allocateBuffers();
    void releaseBuffers() noexcept;

    void forwardAsync(
        void* d_inputG_Buffers,
        float* d_outputLogits,
        cudaStream_t stream);

    void forwardAsync(cudaStream_t stream);

    void* deviceInput() const noexcept { return d_input_; }
    float* deviceOutput() const noexcept { return static_cast<float*>(d_output_); }
    bool hasOwnedBuffers() const noexcept { return d_input_ != nullptr && d_output_ != nullptr; }

private:
    struct TrtObjectDeleter {
        template <typename T>
        void operator()(T* ptr) const noexcept {
            delete ptr;
        }
    };

    template <typename T>
    using TrtPtr = std::unique_ptr<T, TrtObjectDeleter>;

    static std::vector<char> readBinaryFile(const std::string& path);
    static const char* severityToString(Severity severity) noexcept;

    void validateEngineContract() const;
    void validateContextShapes() const;
    void bindAndEnqueue(void* dInput, void* dOutput, cudaStream_t stream);

    TrtPtr<nvinfer1::IRuntime> runtime_;
    TrtPtr<nvinfer1::ICudaEngine> engine_;
    TrtPtr<nvinfer1::IExecutionContext> context_;

    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
};

#define RAY_BUDGET_CHECK_CUDA(expr) \
    RayBudgetInferenceEngine::checkCuda((expr), #expr, __FILE__, __LINE__)
