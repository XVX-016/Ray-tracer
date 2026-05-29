#pragma once

#include "allocation_converter.h"
#include "inference_harness.h"

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iosfwd>
#include <string>
#include <vector>

namespace raybudget {

struct ProfileFrameInputs {
    void* dGBufferTensor = nullptr;
    float* dLogits = nullptr;
    TraceRaysIndirectCommandKHR* dIndirectCommand = nullptr;
    uint8_t* dTileBudgetMap = nullptr;
    uint32_t* dTotalRayCounter = nullptr;
    cudaStream_t cudaStream = nullptr;
};

struct VulkanTimestampContext {
    VkDevice device = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkQueryPool queryPool = VK_NULL_HANDLE;
    float timestampPeriodNs = 1.0f;
    uint32_t timestampValidBits = 64;
    uint32_t frameSlot = 0;
    uint32_t queriesPerFrame = 4;
};

struct FrameTimingMs {
    double inferenceMs = 0.0;
    double converterMs = 0.0;
    double rayTraceMs = 0.0;
    double totalMeasuredMs = 0.0;
};

struct RollingStats {
    size_t sampleCount = 0;
    double meanMs = 0.0;
    double p99Ms = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
};

struct ProfileSummary {
    RollingStats inference;
    RollingStats converter;
    RollingStats rayTrace;
    RollingStats total;
    double meanSavingsVs4SppMs = 0.0;
    double meanSavingsVs8SppMs = 0.0;
    double savingsVs4SppPercent = 0.0;
    double savingsVs8SppPercent = 0.0;
};

constexpr double kUniform4SppBaselineMs = 4.0;
constexpr double kUniform8SppBaselineMs = 8.0;

class NvtxScope final {
public:
    explicit NvtxScope(const char* name) noexcept;
    ~NvtxScope() noexcept;
    NvtxScope(const NvtxScope&) = delete;
    NvtxScope& operator=(const NvtxScope&) = delete;
};

class CudaEventPair final {
public:
    CudaEventPair();
    ~CudaEventPair() noexcept;
    CudaEventPair(const CudaEventPair&) = delete;
    CudaEventPair& operator=(const CudaEventPair&) = delete;
    CudaEventPair(CudaEventPair&& other) noexcept;
    CudaEventPair& operator=(CudaEventPair&& other) noexcept;

    void recordStart(cudaStream_t stream);
    void recordStop(cudaStream_t stream);
    bool isReady() const;
    float elapsedMs() const;

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

class RollingMetricWindow final {
public:
    explicit RollingMetricWindow(size_t capacity = 1000);
    void push(double valueMs);
    RollingStats stats() const;
    size_t size() const noexcept { return values_.size(); }

private:
    size_t capacity_;
    std::deque<double> values_;
};

class MetricsCsvLogger final {
public:
    explicit MetricsCsvLogger(const std::string& csvPath);
    ~MetricsCsvLogger();
    void writeFrame(uint64_t frameIndex, const FrameTimingMs& timing);
    void flush();

private:
    std::ofstream out_;
};

class RayBudgetProfileHarness final {
public:
    RayBudgetProfileHarness(
        RayBudgetInferenceEngine& inferenceEngine,
        double uniform4SppBaselineMs = kUniform4SppBaselineMs,
        double uniform8SppBaselineMs = kUniform8SppBaselineMs,
        size_t rollingWindow = 1000);
    ~RayBudgetProfileHarness() noexcept;

    RayBudgetProfileHarness(const RayBudgetProfileHarness&) = delete;
    RayBudgetProfileHarness& operator=(const RayBudgetProfileHarness&) = delete;

    void beginFrame(uint64_t frameIndex) noexcept;
    void endFrameNvtx() noexcept;

    void recordVulkanGBufferExportBegin();
    void recordVulkanGBufferExportEnd();
    void recordCudaInteropSyncBegin();
    void recordCudaInteropSyncEnd();
    void recordVulkanIndirectRayTraceBegin();
    void recordVulkanIndirectRayTraceEnd();

    void runInferenceAndConverter(const ProfileFrameInputs& inputs);

    void writeRayTraceTimestampBegin(const VulkanTimestampContext& ctx);
    void writeRayTraceTimestampEnd(const VulkanTimestampContext& ctx);
    void resetTimestampQueriesForFrame(const VulkanTimestampContext& ctx);
    bool collectCompletedFrame(
        const VulkanTimestampContext* timestampCtx,
        FrameTimingMs* outTiming);

    ProfileSummary summary() const;
    void writeSummary(std::ostream& out) const;
    void attachCsvLogger(const std::string& csvPath);

    static VkQueryPool createTimestampQueryPool(
        VkDevice device,
        uint32_t framesInFlight,
        uint32_t queriesPerFrame = 4);
    static float timestampPeriodNs(VkPhysicalDevice physicalDevice);
    static uint32_t timestampValidBits(
        VkPhysicalDevice physicalDevice,
        uint32_t graphicsQueueFamily);
    static double timestampsToMilliseconds(
        uint64_t begin,
        uint64_t end,
        float timestampPeriodNs,
        uint32_t validBits);

private:
    struct PendingFrame {
        uint64_t frameIndex = 0;
        CudaEventPair inference;
        CudaEventPair converter;
    };

    RayBudgetInferenceEngine& inferenceEngine_;
    double uniform4SppBaselineMs_;
    double uniform8SppBaselineMs_;
    uint64_t activeFrameIndex_ = 0;

    std::vector<const char*> activeNvtxRanges_;
    std::deque<PendingFrame> pending_;

    RollingMetricWindow inferenceWindow_;
    RollingMetricWindow converterWindow_;
    RollingMetricWindow rayTraceWindow_;
    RollingMetricWindow totalWindow_;

    MetricsCsvLogger* csvLogger_ = nullptr;
};

}  // namespace raybudget
