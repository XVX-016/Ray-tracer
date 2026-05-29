/*
    CUDA converter from TensorRT tile logits to Vulkan indirect ray command.

    Converter input contract:
        logits [120, 68, 6] float32

    The Phase 2 PyTorch model naturally emits [1, 6, 120, 68]. For this final
    interop pass, prefer folding a TensorRT shuffle/transpose into the engine so
    this CUDA kernel receives tile-major NHWC logits. That layout makes each
    tile's six class scores contiguous and keeps one warp block fully local.

    This kernel emits one VkTraceRaysIndirectCommandKHR-compatible struct.
    The class argmax determines how many rays/samples each 16x16 tile should
    receive. Standard vkCmdTraceRaysIndirectKHR only supports a single global
    dispatch size, so this converter writes:

        width  = sum(tile_budget_class_to_ray_count[argmax(tile)])
        height = 1
        depth  = 1

    A production raygen shader consumes an additional tile allocation map when
    width is interpreted as a compacted ray-work stream. If instead your
    implementation traces one raygen invocation per screen pixel, replace the
    accumulation with width=1920, height=1080, depth=1 and consume the per-tile
    budget map in shader code.
*/

#include "allocation_converter.h"

namespace raybudget {

__device__ __constant__ int kClassToRayCount[kClasses] = {0, 1, 2, 4, 8, 16};

__global__ void GenerateIndirectCommands(
    const float* __restrict__ logits,
    TraceRaysIndirectCommandKHR* __restrict__ indirectCommand,
    uint8_t* __restrict__ tileRayBudgetMap,
    uint32_t* __restrict__ totalRayCounter) {
    const int tileLinear = static_cast<int>(blockIdx.x);
    if (tileLinear >= kNumTiles) {
        return;
    }

    const int lane = static_cast<int>(threadIdx.x);
    if (lane >= 32) {
        return;
    }

    float score = -3.402823466e+38F;
    int cls = lane;
    if (lane < kClasses) {
        const int offset = tileLinear * kClasses + lane;
        score = logits[offset];
    }

    unsigned mask = __ballot_sync(0xffffffffu, lane < kClasses);

    for (int step = 16; step > 0; step >>= 1) {
        const float otherScore = __shfl_down_sync(mask, score, step);
        const int otherClass = __shfl_down_sync(mask, cls, step);
        if (otherScore > score || (otherScore == score && otherClass > cls)) {
            score = otherScore;
            cls = otherClass;
        }
    }

    if (lane == 0) {
        const uint8_t rays = static_cast<uint8_t>(kClassToRayCount[cls]);
        tileRayBudgetMap[tileLinear] = rays;
        atomicAdd(totalRayCounter, static_cast<uint32_t>(rays));
    }

    if (tileLinear == 0 && lane == 0) {
        // One block initializes the indirect command before accumulation is read
        // by FinalizeIndirectCommand. This kernel should be launched before the
        // finalize kernel on the same CUDA stream.
        indirectCommand->width = 0;
        indirectCommand->height = 1;
        indirectCommand->depth = 1;
    }
}

__global__ void FinalizeIndirectCommand(
    TraceRaysIndirectCommandKHR* __restrict__ indirectCommand,
    const uint32_t* __restrict__ totalRayCounter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        indirectCommand->width = *totalRayCounter;
        indirectCommand->height = 1;
        indirectCommand->depth = 1;
    }
}

cudaError_t LaunchGenerateIndirectCommands(
    const float* dLogits,
    TraceRaysIndirectCommandKHR* dIndirectCommand,
    uint8_t* dTileRayBudgetMap,
    uint32_t* dTotalRayCounter,
    cudaStream_t stream) {
    if (dLogits == nullptr ||
        dIndirectCommand == nullptr ||
        dTileRayBudgetMap == nullptr ||
        dTotalRayCounter == nullptr) {
        return cudaErrorInvalidValue;
    }

    cudaMemsetAsync(dTotalRayCounter, 0, sizeof(uint32_t), stream);
    GenerateIndirectCommands<<<kNumTiles, 32, 0, stream>>>(
        dLogits,
        dIndirectCommand,
        dTileRayBudgetMap,
        dTotalRayCounter);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    FinalizeIndirectCommand<<<1, 1, 0, stream>>>(dIndirectCommand, dTotalRayCounter);
    return cudaGetLastError();
}

}  // namespace raybudget
