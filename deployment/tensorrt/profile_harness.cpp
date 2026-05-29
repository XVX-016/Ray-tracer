#include "profile_harness.h"

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace raybudget {

namespace {

#define RB_PROFILE_CHECK_CUDA(expr) checkCudaLocal((expr), #expr, __FILE__, __LINE__)
#define RB_PROFILE_CHECK_VK(expr) checkVkLocal((expr), #expr, __FILE__, __LINE__)

void checkCudaLocal(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA failure: " << expr << " at " << file << ":" << line
            << " : " << cudaGetErrorString(status)
            << " (" << static_cast<int>(status) << ")";
        throw std::runtime_error(oss.str());
    }
}

void checkVkLocal(VkResult status, const char* expr, const char* file, int line) {
    if (status != VK_SUCCESS) {
        std::ostringstream oss;
        oss << "Vulkan failure: " << expr << " at " << file << ":" << line
            << " VkResult=" << static_cast<int>(status);
        throw std::runtime_error(oss.str());
    }
}

RollingStats computeStats(const std::deque<double>& values) {
    RollingStats out{};
    out.sampleCount = values.size();
    if (values.empty()) {
        return out;
    }

    std::vector<double> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    out.meanMs = sum / static_cast<double>(values.size());
    out.minMs = sorted.front();
    out.maxMs = sorted.back();

    const size_t p99Index = static_cast<size_t>(
        std::ceil(0.99 * static_cast<double>(sorted.size())) - 1.0);
    out.p99Ms = sorted[std::min(p99Index, sorted.size() - 1)];
    return out;
}

void writeTimestamp(VkCommandBuffer cmd, VkQueryPool pool, uint32_t queryIndex) {
    vkCmdWriteTimestamp2(
        cmd,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        pool,
        queryIndex);
}

}  // namespace

NvtxScope::NvtxScope(const char* name) noexcept {
    nvtxRangePushA(name);
}

NvtxScope::~NvtxScope() noexcept {
    nvtxRangePop();
}

CudaEventPair::CudaEventPair() {
    RB_PROFILE_CHECK_CUDA(cudaEventCreateWithFlags(&start_, cudaEventDefault));
    RB_PROFILE_CHECK_CUDA(cudaEventCreateWithFlags(&stop_, cudaEventDefault));
}

CudaEventPair::~CudaEventPair() noexcept {
    if (start_ != nullptr) {
        cudaEventDestroy(start_);
        start_ = nullptr;
    }
    if (stop_ != nullptr) {
        cudaEventDestroy(stop_);
        stop_ = nullptr;
    }
}

CudaEventPair::CudaEventPair(CudaEventPair&& other) noexcept
    : start_(other.start_), stop_(other.stop_) {
    other.start_ = nullptr;
    other.stop_ = nullptr;
}

CudaEventPair& CudaEventPair::operator=(CudaEventPair&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (start_ != nullptr) {
        cudaEventDestroy(start_);
    }
    if (stop_ != nullptr) {
        cudaEventDestroy(stop_);
    }
    start_ = other.start_;
    stop_ = other.stop_;
    other.start_ = nullptr;
    other.stop_ = nullptr;
    return *this;
}

void CudaEventPair::recordStart(cudaStream_t stream) {
    RB_PROFILE_CHECK_CUDA(cudaEventRecord(start_, stream));
}

void CudaEventPair::recordStop(cudaStream_t stream) {
    RB_PROFILE_CHECK_CUDA(cudaEventRecord(stop_, stream));
}

bool CudaEventPair::isReady() const {
    const cudaError_t status = cudaEventQuery(stop_);
    if (status == cudaSuccess) {
        return true;
    }
    if (status == cudaErrorNotReady) {
        return false;
    }
    RB_PROFILE_CHECK_CUDA(status);
    return false;
}

float CudaEventPair::elapsedMs() const {
    float ms = 0.0f;
    RB_PROFILE_CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

RollingMetricWindow::RollingMetricWindow(size_t capacity)
    : capacity_(capacity == 0 ? 1 : capacity) {}

void RollingMetricWindow::push(double valueMs) {
    if (!std::isfinite(valueMs)) {
        return;
    }
    values_.push_back(valueMs);
    while (values_.size() > capacity_) {
        values_.pop_front();
    }
}

RollingStats RollingMetricWindow::stats() const {
    return computeStats(values_);
}

MetricsCsvLogger::MetricsCsvLogger(const std::string& csvPath)
    : out_(csvPath, std::ios::out | std::ios::trunc) {
    if (!out_) {
        throw std::runtime_error("Failed to open metrics CSV: " + csvPath);
    }
    out_ << "frame,inference_ms,converter_ms,ray_trace_ms,total_measured_ms\n";
}

MetricsCsvLogger::~MetricsCsvLogger() {
    flush();
}

void MetricsCsvLogger::writeFrame(uint64_t frameIndex, const FrameTimingMs& timing) {
    out_ << frameIndex << ','
         << std::fixed << std::setprecision(6)
         << timing.inferenceMs << ','
         << timing.converterMs << ','
         << timing.rayTraceMs << ','
         << timing.totalMeasuredMs << '\n';
}

void MetricsCsvLogger::flush() {
    if (out_) {
        out_.flush();
    }
}

RayBudgetProfileHarness::RayBudgetProfileHarness(
    RayBudgetInferenceEngine& inferenceEngine,
    double uniform4SppBaselineMs,
    double uniform8SppBaselineMs,
    size_t rollingWindow)
    : inferenceEngine_(inferenceEngine),
      uniform4SppBaselineMs_(uniform4SppBaselineMs),
      uniform8SppBaselineMs_(uniform8SppBaselineMs),
      inferenceWindow_(rollingWindow),
      converterWindow_(rollingWindow),
      rayTraceWindow_(rollingWindow),
      totalWindow_(rollingWindow) {
    if (uniform4SppBaselineMs_ <= 0.0 || uniform8SppBaselineMs_ <= 0.0) {
        throw std::invalid_argument("Uniform baseline timings must be positive milliseconds");
    }
}

RayBudgetProfileHarness::~RayBudgetProfileHarness() noexcept {
    while (!activeNvtxRanges_.empty()) {
        nvtxRangePop();
        activeNvtxRanges_.pop_back();
    }
    delete csvLogger_;
    csvLogger_ = nullptr;
}

void RayBudgetProfileHarness::beginFrame(uint64_t frameIndex) noexcept {
    activeFrameIndex_ = frameIndex;
    nvtxRangePushA("Frame");
    activeNvtxRanges_.push_back("Frame");
}

void RayBudgetProfileHarness::endFrameNvtx() noexcept {
    if (!activeNvtxRanges_.empty()) {
        nvtxRangePop();
        activeNvtxRanges_.pop_back();
    }
}

void RayBudgetProfileHarness::recordVulkanGBufferExportBegin() {
    nvtxRangePushA("Vulkan_GBuffer_Export");
    activeNvtxRanges_.push_back("Vulkan_GBuffer_Export");
}

void RayBudgetProfileHarness::recordVulkanGBufferExportEnd() {
    if (activeNvtxRanges_.empty()) {
        throw std::runtime_error("NVTX range stack underflow at Vulkan_GBuffer_Export end");
    }
    nvtxRangePop();
    activeNvtxRanges_.pop_back();
}

void RayBudgetProfileHarness::recordCudaInteropSyncBegin() {
    nvtxRangePushA("CUDA_Interop_Sync");
    activeNvtxRanges_.push_back("CUDA_Interop_Sync");
}

void RayBudgetProfileHarness::recordCudaInteropSyncEnd() {
    if (activeNvtxRanges_.empty()) {
        throw std::runtime_error("NVTX range stack underflow at CUDA_Interop_Sync end");
    }
    nvtxRangePop();
    activeNvtxRanges_.pop_back();
}

void RayBudgetProfileHarness::recordVulkanIndirectRayTraceBegin() {
    nvtxRangePushA("Vulkan_Indirect_RayTrace");
    activeNvtxRanges_.push_back("Vulkan_Indirect_RayTrace");
}

void RayBudgetProfileHarness::recordVulkanIndirectRayTraceEnd() {
    if (activeNvtxRanges_.empty()) {
        throw std::runtime_error("NVTX range stack underflow at Vulkan_Indirect_RayTrace end");
    }
    nvtxRangePop();
    activeNvtxRanges_.pop_back();
}

void RayBudgetProfileHarness::runInferenceAndConverter(const ProfileFrameInputs& inputs) {
    if (inputs.cudaStream == nullptr) {
        throw std::invalid_argument("ProfileFrameInputs.cudaStream is null");
    }

    PendingFrame pending{};
    pending.frameIndex = activeFrameIndex_;

    {
        NvtxScope inferenceScope("TensorRT_Inference");
        pending.inference.recordStart(inputs.cudaStream);
        inferenceEngine_.forwardAsync(inputs.dGBufferTensor, inputs.dLogits, inputs.cudaStream);
        pending.inference.recordStop(inputs.cudaStream);
    }
    RB_PROFILE_CHECK_CUDA(cudaGetLastError());

    {
        NvtxScope converterScope("CUDA_Indirect_Conversion");
        pending.converter.recordStart(inputs.cudaStream);
        const cudaError_t status = LaunchGenerateIndirectCommands(
            inputs.dLogits,
            inputs.dIndirectCommand,
            inputs.dTileBudgetMap,
            inputs.dTotalRayCounter,
            inputs.cudaStream);
        RB_PROFILE_CHECK_CUDA(status);
        pending.converter.recordStop(inputs.cudaStream);
    }
    RB_PROFILE_CHECK_CUDA(cudaGetLastError());

    pending_.push_back(std::move(pending));
}

void RayBudgetProfileHarness::writeRayTraceTimestampBegin(
    const VulkanTimestampContext& ctx) {
    const uint32_t base = ctx.frameSlot * ctx.queriesPerFrame;
    writeTimestamp(ctx.commandBuffer, ctx.queryPool, base + 0);
}

void RayBudgetProfileHarness::resetTimestampQueriesForFrame(
    const VulkanTimestampContext& ctx) {
    const uint32_t base = ctx.frameSlot * ctx.queriesPerFrame;
    vkCmdResetQueryPool(ctx.commandBuffer, ctx.queryPool, base, ctx.queriesPerFrame);
}

void RayBudgetProfileHarness::writeRayTraceTimestampEnd(
    const VulkanTimestampContext& ctx) {
    const uint32_t base = ctx.frameSlot * ctx.queriesPerFrame;
    writeTimestamp(ctx.commandBuffer, ctx.queryPool, base + 1);
}

bool RayBudgetProfileHarness::collectCompletedFrame(
    const VulkanTimestampContext* timestampCtx,
    FrameTimingMs* outTiming) {
    if (outTiming == nullptr) {
        throw std::invalid_argument("outTiming must not be null");
    }
    if (pending_.empty()) {
        return false;
    }

    PendingFrame& front = pending_.front();
    if (!front.inference.isReady() || !front.converter.isReady()) {
        return false;
    }

    FrameTimingMs timing{};
    timing.inferenceMs = static_cast<double>(front.inference.elapsedMs());
    timing.converterMs = static_cast<double>(front.converter.elapsedMs());

    if (timestampCtx != nullptr && timestampCtx->device != VK_NULL_HANDLE) {
        const uint32_t base = timestampCtx->frameSlot * timestampCtx->queriesPerFrame;
        uint64_t timestamps[2] = {};
        const VkResult result = vkGetQueryPoolResults(
            timestampCtx->device,
            timestampCtx->queryPool,
            base,
            2,
            sizeof(timestamps),
            timestamps,
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT);

        if (result == VK_SUCCESS) {
            timing.rayTraceMs = timestampsToMilliseconds(
                timestamps[0],
                timestamps[1],
                timestampCtx->timestampPeriodNs,
                timestampCtx->timestampValidBits);
        } else if (result != VK_NOT_READY) {
            RB_PROFILE_CHECK_VK(result);
        } else {
            return false;
        }
    }

    timing.totalMeasuredMs = timing.inferenceMs + timing.converterMs + timing.rayTraceMs;

    inferenceWindow_.push(timing.inferenceMs);
    converterWindow_.push(timing.converterMs);
    rayTraceWindow_.push(timing.rayTraceMs);
    totalWindow_.push(timing.totalMeasuredMs);

    if (csvLogger_ != nullptr) {
        csvLogger_->writeFrame(front.frameIndex, timing);
    }

    *outTiming = timing;
    pending_.pop_front();
    return true;
}

ProfileSummary RayBudgetProfileHarness::summary() const {
    ProfileSummary out{};
    out.inference = inferenceWindow_.stats();
    out.converter = converterWindow_.stats();
    out.rayTrace = rayTraceWindow_.stats();
    out.total = totalWindow_.stats();

    out.meanSavingsVs4SppMs = uniform4SppBaselineMs_ - out.total.meanMs;
    out.meanSavingsVs8SppMs = uniform8SppBaselineMs_ - out.total.meanMs;
    out.savingsVs4SppPercent =
        (out.meanSavingsVs4SppMs / uniform4SppBaselineMs_) * 100.0;
    out.savingsVs8SppPercent =
        (out.meanSavingsVs8SppMs / uniform8SppBaselineMs_) * 100.0;
    return out;
}

void RayBudgetProfileHarness::writeSummary(std::ostream& out) const {
    const ProfileSummary s = summary();
    auto writeStats = [&out](const char* name, const RollingStats& stats) {
        out << name
            << " samples=" << stats.sampleCount
            << " mean_ms=" << std::fixed << std::setprecision(6) << stats.meanMs
            << " p99_ms=" << stats.p99Ms
            << " min_ms=" << stats.minMs
            << " max_ms=" << stats.maxMs
            << '\n';
    };

    writeStats("TensorRT_Inference", s.inference);
    writeStats("CUDA_Indirect_Conversion", s.converter);
    writeStats("Vulkan_Indirect_RayTrace", s.rayTrace);
    writeStats("Total_Inference_Converter_RayTrace", s.total);
    out << "Savings_vs_uniform_4spp_ms=" << s.meanSavingsVs4SppMs
        << " percent=" << s.savingsVs4SppPercent << '\n';
    out << "Savings_vs_uniform_8spp_ms=" << s.meanSavingsVs8SppMs
        << " percent=" << s.savingsVs8SppPercent << '\n';
}

void RayBudgetProfileHarness::attachCsvLogger(const std::string& csvPath) {
    delete csvLogger_;
    csvLogger_ = new MetricsCsvLogger(csvPath);
}

VkQueryPool RayBudgetProfileHarness::createTimestampQueryPool(
    VkDevice device,
    uint32_t framesInFlight,
    uint32_t queriesPerFrame) {
    if (device == VK_NULL_HANDLE) {
        throw std::invalid_argument("VkDevice is null");
    }
    if (framesInFlight == 0 || queriesPerFrame < 2) {
        throw std::invalid_argument("Invalid query pool dimensions");
    }

    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = framesInFlight * queriesPerFrame;

    VkQueryPool pool = VK_NULL_HANDLE;
    RB_PROFILE_CHECK_VK(vkCreateQueryPool(device, &info, nullptr, &pool));
    return pool;
}

float RayBudgetProfileHarness::timestampPeriodNs(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    return props.limits.timestampPeriod;
}

uint32_t RayBudgetProfileHarness::timestampValidBits(
    VkPhysicalDevice physicalDevice,
    uint32_t graphicsQueueFamily) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
    if (graphicsQueueFamily >= count) {
        throw std::out_of_range("graphicsQueueFamily out of range");
    }
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, families.data());
    return families[graphicsQueueFamily].timestampValidBits;
}

double RayBudgetProfileHarness::timestampsToMilliseconds(
    uint64_t begin,
    uint64_t end,
    float timestampPeriodNs,
    uint32_t validBits) {
    if (timestampPeriodNs <= 0.0f) {
        throw std::invalid_argument("timestampPeriodNs must be positive");
    }
    if (validBits == 0) {
        return 0.0;
    }

    uint64_t mask = std::numeric_limits<uint64_t>::max();
    if (validBits < 64) {
        mask = (uint64_t{1} << validBits) - 1u;
    }

    begin &= mask;
    end &= mask;

    const uint64_t deltaTicks =
        (end >= begin) ? (end - begin) : ((mask - begin) + end + 1u);
    const double ns = static_cast<double>(deltaTicks) * static_cast<double>(timestampPeriodNs);
    return ns / 1.0e6;
}

/*
Build and Nsight Systems usage
------------------------------

1. Make sure CUDA, Vulkan SDK, TensorRT, and NVTX3 headers are visible. Recent
   CUDA toolkits ship nvtx3/nvToolsExt.h under the CUDA include directory.

2. Configure:

      cmake -S deployment/tensorrt -B build/tensorrt \
        -DTENSORRT_ROOT=/path/to/TensorRT

3. Build:

      cmake --build build/tensorrt --config Release

4. Capture an Nsight Systems timeline around your application:

      nsys profile \
        --trace=cuda,nvtx,vulkan,osrt \
        --capture-range=nvtx \
        --capture-range-end=stop \
        --output=ray_budget_phase5 \
        ./your_renderer

   Open ray_budget_phase5.nsys-rep in Nsight Systems. The expected nested
   ranges are:

      Frame
        Vulkan_GBuffer_Export
        CUDA_Interop_Sync
        TensorRT_Inference
        CUDA_Indirect_Conversion
        Vulkan_Indirect_RayTrace

5. Interpretation:
   CUDA event timings report GPU elapsed time for TensorRT and converter work
   without CPU synchronization in the frame. Vulkan timestamp queries report
   GPU timeline duration for ray tracing. collectCompletedFrame() harvests only
   completed event/query data and returns false while work is still in flight.
*/

}  // namespace raybudget
