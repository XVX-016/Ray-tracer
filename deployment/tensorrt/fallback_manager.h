#pragma once

#include "allocation_converter.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace raybudget {

constexpr float kDefaultTimeoutMs = 1.5f;
constexpr uint8_t kFallbackUniformSpp = 4;
constexpr int kTemporalHistoryFrames = 3;

enum FallbackReason : uint32_t {
    kFallbackNone = 0,
    kFallbackTimeout = 1u << 0,
    kFallbackOodLuminance = 1u << 1,
    kFallbackOodMotion = 1u << 2,
    kFallbackRuntimeError = 1u << 3,
};

struct OodThresholds {
    float luminanceDelta = 0.35f;
    float meanMotionPixels = 48.0f;
};

struct FallbackDeviceState {
    uint32_t fallbackRequested;
    uint32_t fallbackReasonMask;
    uint32_t historyValidFrames;
    uint32_t frameIndex;
    float previousMeanLuminance;
    float currentMeanLuminance;
    float currentMeanMotion;
};

struct FallbackFrameInputs {
    const float* dLogits = nullptr;       // [120, 68, 6], float32
    const float* dLuminance = nullptr;    // [1080, 1920], float32
    const float* dMotionXY = nullptr;     // [1080, 1920, 2], float32

    TraceRaysIndirectCommandKHR* dIndirectCommand = nullptr;
    uint8_t* dTileBudgetMap = nullptr;    // [120 * 68], uint8_t
    uint32_t* dTotalRayCounter = nullptr;
};

class FallbackManager final {
public:
    explicit FallbackManager(
        float timeoutMs = kDefaultTimeoutMs,
        uint8_t fallbackSpp = kFallbackUniformSpp,
        OodThresholds thresholds = {});
    ~FallbackManager() noexcept;

    FallbackManager(const FallbackManager&) = delete;
    FallbackManager& operator=(const FallbackManager&) = delete;

    void resetAsync(cudaStream_t stream);

    void markInferenceStart(cudaStream_t stream);
    void markInferenceStop(cudaStream_t stream);

    /*
        Watchdog semantics:
        - inferenceTimedOutOrErrored() returns true when the stop event is not
          ready. Use it from a CPU frame watchdog only if the fallback writes to
          a separate per-frame indirect buffer, or after the original producer
          path has been excluded from consuming this frame's output.
        - inferenceExceededBudget() returns true only after the stop event is
          complete and elapsed GPU time exceeds timeoutMs().

        This avoids host readback bubbles; both methods use cudaEventQuery.
    */
    bool inferenceTimedOutOrErrored();
    bool inferenceExceededBudget();

    cudaError_t analyzeOodAsync(
        const FallbackFrameInputs& inputs,
        cudaStream_t stream);

    cudaError_t generateSafeAllocationAsync(
        const FallbackFrameInputs& inputs,
        cudaStream_t stream);

    cudaError_t forceFallbackAsync(
        const FallbackFrameInputs& inputs,
        FallbackReason reason,
        cudaStream_t stream);

    FallbackDeviceState* deviceState() const noexcept { return dState_; }
    float timeoutMs() const noexcept { return timeoutMs_; }

private:
    void allocate();
    void release() noexcept;
    void checkCuda(cudaError_t status, const char* expr, const char* file, int line) const;

    float timeoutMs_;
    uint8_t fallbackSpp_;
    OodThresholds thresholds_;

    cudaEvent_t inferenceStart_ = nullptr;
    cudaEvent_t inferenceStop_ = nullptr;

    FallbackDeviceState* dState_ = nullptr;
    uint8_t* dHistory_ = nullptr;         // [3, 8160]
    uint32_t* dScratchCounters_ = nullptr; // [4]: luminance sum bits, motion sum bits, count, flags
};

cudaError_t LaunchAnalyzeOodFrame(
    const float* dLuminance,
    const float* dMotionXY,
    FallbackDeviceState* dState,
    uint32_t* dScratchCounters,
    OodThresholds thresholds,
    cudaStream_t stream);

cudaError_t LaunchGenerateResilientCommands(
    const float* dLogits,
    TraceRaysIndirectCommandKHR* dIndirectCommand,
    uint8_t* dTileRayBudgetMap,
    uint32_t* dTotalRayCounter,
    FallbackDeviceState* dState,
    uint8_t* dHistory,
    uint8_t fallbackSpp,
    cudaStream_t stream);

cudaError_t LaunchForceUniformFallback(
    TraceRaysIndirectCommandKHR* dIndirectCommand,
    uint8_t* dTileRayBudgetMap,
    uint32_t* dTotalRayCounter,
    FallbackDeviceState* dState,
    uint8_t fallbackSpp,
    FallbackReason reason,
    cudaStream_t stream);

}  // namespace raybudget
