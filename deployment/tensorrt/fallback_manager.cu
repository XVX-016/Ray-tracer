#include "fallback_manager.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace raybudget {

namespace {

constexpr int kPixels = 1080 * 1920;
constexpr int kAnalysisThreads = 256;
constexpr int kAnalysisBlocks = 256;
constexpr int kTileThreads = 32;
constexpr int kRayClassCount = 6;

__device__ __constant__ uint8_t kClassToRayCountU8[kRayClassCount] = {0, 1, 2, 4, 8, 16};

__device__ float orderedUintToFloat(uint32_t v) {
    return __uint_as_float(v);
}

__device__ void atomicAddFloatAsUint(uint32_t* address, float value) {
    atomicAdd(reinterpret_cast<float*>(address), value);
}

__global__ void ResetFallbackStateKernel(
    FallbackDeviceState* state,
    uint8_t* history,
    uint32_t* scratch) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        state->fallbackRequested = 0;
        state->fallbackReasonMask = kFallbackNone;
        state->historyValidFrames = 0;
        state->frameIndex = 0;
        state->previousMeanLuminance = 0.0f;
        state->currentMeanLuminance = 0.0f;
        state->currentMeanMotion = 0.0f;
        scratch[0] = 0;
        scratch[1] = 0;
        scratch[2] = 0;
        scratch[3] = 0;
    }
    for (int i = tid; i < kTemporalHistoryFrames * kNumTiles; i += blockDim.x * gridDim.x) {
        history[i] = kFallbackUniformSpp;
    }
}

__global__ void ClearOodScratchKernel(uint32_t* scratch) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        scratch[0] = 0;
        scratch[1] = 0;
        scratch[2] = 0;
        scratch[3] = 0;
    }
}

__global__ void AccumulateOodKernel(
    const float* __restrict__ luminance,
    const float* __restrict__ motionXY,
    uint32_t* __restrict__ scratch) {
    __shared__ float localLuma[kAnalysisThreads];
    __shared__ float localMotion[kAnalysisThreads];
    __shared__ uint32_t localCount[kAnalysisThreads];

    const int tid = threadIdx.x;
    float lumaSum = 0.0f;
    float motionSum = 0.0f;
    uint32_t count = 0;

    for (int idx = blockIdx.x * blockDim.x + tid;
         idx < kPixels;
         idx += gridDim.x * blockDim.x) {
        const float lum = fmaxf(luminance[idx], 0.0f);
        const float mx = motionXY[idx * 2 + 0];
        const float my = motionXY[idx * 2 + 1];
        lumaSum += lum;
        motionSum += sqrtf(mx * mx + my * my);
        ++count;
    }

    localLuma[tid] = lumaSum;
    localMotion[tid] = motionSum;
    localCount[tid] = count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            localLuma[tid] += localLuma[tid + stride];
            localMotion[tid] += localMotion[tid + stride];
            localCount[tid] += localCount[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAddFloatAsUint(&scratch[0], localLuma[0]);
        atomicAddFloatAsUint(&scratch[1], localMotion[0]);
        atomicAdd(&scratch[2], localCount[0]);
    }
}

__global__ void FinalizeOodKernel(
    FallbackDeviceState* state,
    const uint32_t* scratch,
    OodThresholds thresholds) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    const uint32_t count = max(scratch[2], 1u);
    const float meanLum = orderedUintToFloat(scratch[0]) / static_cast<float>(count);
    const float meanMotion = orderedUintToFloat(scratch[1]) / static_cast<float>(count);

    const float prevLum = state->previousMeanLuminance;
    const bool hasHistory = state->historyValidFrames > 0;
    const float lumDelta = hasHistory
        ? fabsf(meanLum - prevLum) / fmaxf(prevLum, 1.0e-3f)
        : 0.0f;

    uint32_t reason = kFallbackNone;
    if (hasHistory && lumDelta > thresholds.luminanceDelta) {
        reason |= kFallbackOodLuminance;
    }
    if (meanMotion > thresholds.meanMotionPixels) {
        reason |= kFallbackOodMotion;
    }

    state->currentMeanLuminance = meanLum;
    state->currentMeanMotion = meanMotion;
    if (reason != kFallbackNone) {
        state->fallbackRequested = 1;
        atomicOr(&state->fallbackReasonMask, reason);
        state->historyValidFrames = 0;
    }
}

__device__ uint8_t argmaxRayCountForTile(
    const float* __restrict__ logits,
    int tileLinear) {
    float bestScore = -3.402823466e+38F;
    int bestClass = 0;
    for (int cls = 0; cls < kRayClassCount; ++cls) {
        const float score = logits[tileLinear * kRayClassCount + cls];
        if (score > bestScore || (score == bestScore && cls > bestClass)) {
            bestScore = score;
            bestClass = cls;
        }
    }
    return kClassToRayCountU8[bestClass];
}

__device__ uint8_t debouncedRayCount(
    uint8_t proposed,
    uint8_t previous,
    uint32_t validFrames) {
    if (validFrames == 0) {
        return proposed;
    }
    if (proposed >= previous) {
        return proposed;
    }
    const uint8_t step = 1;
    const uint8_t decremented = static_cast<uint8_t>(previous > step ? previous - step : 0);
    return proposed > decremented ? proposed : decremented;
}

__global__ void GenerateResilientCommandsKernel(
    const float* __restrict__ logits,
    TraceRaysIndirectCommandKHR* __restrict__ indirectCommand,
    uint8_t* __restrict__ tileRayBudgetMap,
    uint32_t* __restrict__ totalRayCounter,
    FallbackDeviceState* __restrict__ state,
    uint8_t* __restrict__ history,
    uint8_t fallbackSpp) {
    const int tileLinear = blockIdx.x * blockDim.x + threadIdx.x;

    if (tileLinear == 0) {
        indirectCommand->width = 0;
        indirectCommand->height = 1;
        indirectCommand->depth = 1;
    }

    if (tileLinear >= kNumTiles) {
        return;
    }

    const uint32_t frameSlot = state->frameIndex % kTemporalHistoryFrames;
    const uint32_t previousSlot =
        (state->frameIndex + kTemporalHistoryFrames - 1) % kTemporalHistoryFrames;
    const uint32_t historyValid = state->historyValidFrames;
    const bool fallback = state->fallbackRequested != 0;

    uint8_t rays = fallback ? fallbackSpp : argmaxRayCountForTile(logits, tileLinear);
    const uint8_t previous = history[previousSlot * kNumTiles + tileLinear];
    rays = fallback ? fallbackSpp : debouncedRayCount(rays, previous, historyValid);

    tileRayBudgetMap[tileLinear] = rays;
    history[frameSlot * kNumTiles + tileLinear] = rays;
    atomicAdd(totalRayCounter, static_cast<uint32_t>(rays));
}

__global__ void FinalizeResilientCommandKernel(
    TraceRaysIndirectCommandKHR* __restrict__ indirectCommand,
    const uint32_t* __restrict__ totalRayCounter,
    FallbackDeviceState* __restrict__ state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    indirectCommand->width = *totalRayCounter;
    indirectCommand->height = 1;
    indirectCommand->depth = 1;

    if (state->fallbackRequested == 0) {
        state->historyValidFrames = min(state->historyValidFrames + 1u,
                                        static_cast<uint32_t>(kTemporalHistoryFrames));
    }
    state->previousMeanLuminance = state->currentMeanLuminance;
    state->fallbackRequested = 0;
    state->fallbackReasonMask = kFallbackNone;
    state->frameIndex += 1;
}

__global__ void ForceUniformFallbackKernel(
    TraceRaysIndirectCommandKHR* __restrict__ indirectCommand,
    uint8_t* __restrict__ tileRayBudgetMap,
    uint32_t* __restrict__ totalRayCounter,
    FallbackDeviceState* __restrict__ state,
    uint8_t fallbackSpp,
    FallbackReason reason) {
    const int tileLinear = blockIdx.x * blockDim.x + threadIdx.x;
    if (tileLinear < kNumTiles) {
        tileRayBudgetMap[tileLinear] = fallbackSpp;
    }
    if (tileLinear == 0) {
        *totalRayCounter = static_cast<uint32_t>(fallbackSpp) * kNumTiles;
        indirectCommand->width = *totalRayCounter;
        indirectCommand->height = 1;
        indirectCommand->depth = 1;
        state->fallbackRequested = 1;
        atomicOr(&state->fallbackReasonMask, static_cast<uint32_t>(reason));
        state->historyValidFrames = 0;
        state->frameIndex += 1;
    }
}

}  // namespace

FallbackManager::FallbackManager(
    float timeoutMs,
    uint8_t fallbackSpp,
    OodThresholds thresholds)
    : timeoutMs_(timeoutMs),
      fallbackSpp_(fallbackSpp),
      thresholds_(thresholds) {
    if (timeoutMs_ <= 0.0f) {
        throw std::invalid_argument("FallbackManager timeout must be positive");
    }
    allocate();
}

FallbackManager::~FallbackManager() noexcept {
    release();
}

void FallbackManager::allocate() {
    checkCuda(cudaEventCreateWithFlags(&inferenceStart_, cudaEventDefault),
              "cudaEventCreateWithFlags(start)", __FILE__, __LINE__);
    checkCuda(cudaEventCreateWithFlags(&inferenceStop_, cudaEventDefault),
              "cudaEventCreateWithFlags(stop)", __FILE__, __LINE__);
    checkCuda(cudaMalloc(&dState_, sizeof(FallbackDeviceState)),
              "cudaMalloc(dState_)", __FILE__, __LINE__);
    checkCuda(cudaMalloc(&dHistory_, kTemporalHistoryFrames * kNumTiles * sizeof(uint8_t)),
              "cudaMalloc(dHistory_)", __FILE__, __LINE__);
    checkCuda(cudaMalloc(&dScratchCounters_, 4 * sizeof(uint32_t)),
              "cudaMalloc(dScratchCounters_)", __FILE__, __LINE__);
}

void FallbackManager::release() noexcept {
    if (inferenceStart_ != nullptr) {
        cudaEventDestroy(inferenceStart_);
        inferenceStart_ = nullptr;
    }
    if (inferenceStop_ != nullptr) {
        cudaEventDestroy(inferenceStop_);
        inferenceStop_ = nullptr;
    }
    if (dState_ != nullptr) {
        cudaFree(dState_);
        dState_ = nullptr;
    }
    if (dHistory_ != nullptr) {
        cudaFree(dHistory_);
        dHistory_ = nullptr;
    }
    if (dScratchCounters_ != nullptr) {
        cudaFree(dScratchCounters_);
        dScratchCounters_ = nullptr;
    }
}

void FallbackManager::checkCuda(
    cudaError_t status,
    const char* expr,
    const char* file,
    int line) const {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA failure: " << expr << " at " << file << ":" << line
            << " : " << cudaGetErrorString(status)
            << " (" << static_cast<int>(status) << ")";
        throw std::runtime_error(oss.str());
    }
}

void FallbackManager::resetAsync(cudaStream_t stream) {
    ResetFallbackStateKernel<<<32, 256, 0, stream>>>(dState_, dHistory_, dScratchCounters_);
    checkCuda(cudaGetLastError(), "ResetFallbackStateKernel", __FILE__, __LINE__);
}

void FallbackManager::markInferenceStart(cudaStream_t stream) {
    checkCuda(cudaEventRecord(inferenceStart_, stream),
              "cudaEventRecord(inferenceStart_)", __FILE__, __LINE__);
}

void FallbackManager::markInferenceStop(cudaStream_t stream) {
    checkCuda(cudaEventRecord(inferenceStop_, stream),
              "cudaEventRecord(inferenceStop_)", __FILE__, __LINE__);
}

bool FallbackManager::inferenceTimedOutOrErrored() {
    const cudaError_t query = cudaEventQuery(inferenceStop_);
    if (query == cudaSuccess) {
        return inferenceExceededBudget();
    }
    if (query == cudaErrorNotReady) {
        return true;
    }
    return true;
}

bool FallbackManager::inferenceExceededBudget() {
    const cudaError_t query = cudaEventQuery(inferenceStop_);
    if (query != cudaSuccess) {
        return false;
    }
    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, inferenceStart_, inferenceStop_),
              "cudaEventElapsedTime(inference)", __FILE__, __LINE__);
    return elapsedMs > timeoutMs_;
}

cudaError_t FallbackManager::analyzeOodAsync(
    const FallbackFrameInputs& inputs,
    cudaStream_t stream) {
    if (inputs.dLuminance == nullptr || inputs.dMotionXY == nullptr) {
        return cudaErrorInvalidValue;
    }
    return LaunchAnalyzeOodFrame(
        inputs.dLuminance,
        inputs.dMotionXY,
        dState_,
        dScratchCounters_,
        thresholds_,
        stream);
}

cudaError_t FallbackManager::generateSafeAllocationAsync(
    const FallbackFrameInputs& inputs,
    cudaStream_t stream) {
    if (inputs.dLogits == nullptr ||
        inputs.dIndirectCommand == nullptr ||
        inputs.dTileBudgetMap == nullptr ||
        inputs.dTotalRayCounter == nullptr) {
        return cudaErrorInvalidValue;
    }
    return LaunchGenerateResilientCommands(
        inputs.dLogits,
        inputs.dIndirectCommand,
        inputs.dTileBudgetMap,
        inputs.dTotalRayCounter,
        dState_,
        dHistory_,
        fallbackSpp_,
        stream);
}

cudaError_t FallbackManager::forceFallbackAsync(
    const FallbackFrameInputs& inputs,
    FallbackReason reason,
    cudaStream_t stream) {
    if (inputs.dIndirectCommand == nullptr ||
        inputs.dTileBudgetMap == nullptr ||
        inputs.dTotalRayCounter == nullptr) {
        return cudaErrorInvalidValue;
    }
    return LaunchForceUniformFallback(
        inputs.dIndirectCommand,
        inputs.dTileBudgetMap,
        inputs.dTotalRayCounter,
        dState_,
        fallbackSpp_,
        reason,
        stream);
}

cudaError_t LaunchAnalyzeOodFrame(
    const float* dLuminance,
    const float* dMotionXY,
    FallbackDeviceState* dState,
    uint32_t* dScratchCounters,
    OodThresholds thresholds,
    cudaStream_t stream) {
    if (dLuminance == nullptr || dMotionXY == nullptr ||
        dState == nullptr || dScratchCounters == nullptr) {
        return cudaErrorInvalidValue;
    }
    ClearOodScratchKernel<<<1, 1, 0, stream>>>(dScratchCounters);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    AccumulateOodKernel<<<kAnalysisBlocks, kAnalysisThreads, 0, stream>>>(
        dLuminance,
        dMotionXY,
        dScratchCounters);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    FinalizeOodKernel<<<1, 1, 0, stream>>>(dState, dScratchCounters, thresholds);
    return cudaGetLastError();
}

cudaError_t LaunchGenerateResilientCommands(
    const float* dLogits,
    TraceRaysIndirectCommandKHR* dIndirectCommand,
    uint8_t* dTileRayBudgetMap,
    uint32_t* dTotalRayCounter,
    FallbackDeviceState* dState,
    uint8_t* dHistory,
    uint8_t fallbackSpp,
    cudaStream_t stream) {
    if (dLogits == nullptr || dIndirectCommand == nullptr ||
        dTileRayBudgetMap == nullptr || dTotalRayCounter == nullptr ||
        dState == nullptr || dHistory == nullptr) {
        return cudaErrorInvalidValue;
    }
    cudaMemsetAsync(dTotalRayCounter, 0, sizeof(uint32_t), stream);
    GenerateResilientCommandsKernel<<<(kNumTiles + 255) / 256, 256, 0, stream>>>(
        dLogits,
        dIndirectCommand,
        dTileRayBudgetMap,
        dTotalRayCounter,
        dState,
        dHistory,
        fallbackSpp);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    FinalizeResilientCommandKernel<<<1, 1, 0, stream>>>(
        dIndirectCommand,
        dTotalRayCounter,
        dState);
    return cudaGetLastError();
}

cudaError_t LaunchForceUniformFallback(
    TraceRaysIndirectCommandKHR* dIndirectCommand,
    uint8_t* dTileRayBudgetMap,
    uint32_t* dTotalRayCounter,
    FallbackDeviceState* dState,
    uint8_t fallbackSpp,
    FallbackReason reason,
    cudaStream_t stream) {
    if (dIndirectCommand == nullptr || dTileRayBudgetMap == nullptr ||
        dTotalRayCounter == nullptr || dState == nullptr) {
        return cudaErrorInvalidValue;
    }
    ForceUniformFallbackKernel<<<(kNumTiles + 255) / 256, 256, 0, stream>>>(
        dIndirectCommand,
        dTileRayBudgetMap,
        dTotalRayCounter,
        dState,
        fallbackSpp,
        reason);
    return cudaGetLastError();
}

}  // namespace raybudget
