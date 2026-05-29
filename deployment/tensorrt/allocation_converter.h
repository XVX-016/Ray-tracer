#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace raybudget {

constexpr int kTileX = 120;
constexpr int kTileY = 68;
constexpr int kNumTiles = kTileX * kTileY;
constexpr int kClasses = 6;

struct alignas(4) TraceRaysIndirectCommandKHR {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
};

static_assert(sizeof(TraceRaysIndirectCommandKHR) == 12,
              "VkTraceRaysIndirectCommandKHR binary layout must be 12 bytes");

cudaError_t LaunchGenerateIndirectCommands(
    const float* dLogits,
    TraceRaysIndirectCommandKHR* dIndirectCommand,
    uint8_t* dTileRayBudgetMap,
    uint32_t* dTotalRayCounter,
    cudaStream_t stream);

}  // namespace raybudget

