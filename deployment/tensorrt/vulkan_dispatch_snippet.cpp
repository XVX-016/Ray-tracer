/*
    Phase 4 dispatch sequencing snippet.

    This file shows the intended frame orchestration. It assumes:
      - G-buffer packing writes a linear external VkBuffer:
            input_gbuffers [1, 7, 1080, 1920] float32
      - TensorRT writes or shuffles logits into an external output buffer:
            output_tile_logits [120, 68, 6] float32
      - allocation_converter.cu writes:
            VkTraceRaysIndirectCommandKHR { width, height, depth }
        into a Vulkan buffer created with:
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

    Vulkan barriers synchronize Vulkan queues only. CUDA/TensorRT interop also
    requires an external semaphore imported into CUDA. The comments mark where
    cudaWaitExternalSemaphoresAsync / cudaSignalExternalSemaphoresAsync should
    be used in the production frame graph.
*/

#include "profile_harness.h"

#include <vulkan/vulkan.h>
#include <cuda_runtime_api.h>

#include <stdexcept>

namespace raybudget {

struct FrameInteropResources {
    VkCommandBuffer graphicsCmd = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue asyncComputeQueue = VK_NULL_HANDLE;

    uint32_t graphicsQueueFamily = 0;
    uint32_t asyncComputeQueueFamily = 0;

    VkBuffer gbufferTensorBuffer = VK_NULL_HANDLE;
    VkBuffer logitsBuffer = VK_NULL_HANDLE;
    VkBuffer tileBudgetBuffer = VK_NULL_HANDLE;
    VkBuffer indirectCommandBuffer = VK_NULL_HANDLE;
    VkDeviceAddress indirectCommandDeviceAddress = 0;

    void* dGBufferTensor = nullptr;
    float* dLogits = nullptr;
    TraceRaysIndirectCommandKHR* dIndirectCommand = nullptr;
    uint8_t* dTileBudgetMap = nullptr;
    uint32_t* dTotalRayCounter = nullptr;

    cudaStream_t cudaStream = nullptr;

    VkStridedDeviceAddressRegionKHR raygenSbt{};
    VkStridedDeviceAddressRegionKHR missSbt{};
    VkStridedDeviceAddressRegionKHR hitSbt{};
    VkStridedDeviceAddressRegionKHR callableSbt{};
};

using PFN_vkCmdTraceRaysIndirectKHRChecked = PFN_vkCmdTraceRaysIndirectKHR;

void insertGraphicsToCudaReleaseBarrier(VkCommandBuffer cmd, const FrameInteropResources& r) {
    VkBufferMemoryBarrier2 barriers[2]{};

    // G-buffer tensor: Vulkan shader writes -> CUDA/TensorRT reads.
    barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barriers[0].srcStageMask =
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    barriers[0].dstAccessMask = VK_ACCESS_2_NONE;
    barriers[0].srcQueueFamilyIndex = r.graphicsQueueFamily;
    barriers[0].dstQueueFamilyIndex = r.asyncComputeQueueFamily;
    barriers[0].buffer = r.gbufferTensorBuffer;
    barriers[0].offset = 0;
    barriers[0].size = VK_WHOLE_SIZE;

    // Indirect/logit buffers are consumed by CUDA this frame. If they were
    // read by Vulkan in the previous frame, release them before CUDA writes.
    barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    barriers[1].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    barriers[1].dstAccessMask = VK_ACCESS_2_NONE;
    barriers[1].srcQueueFamilyIndex = r.graphicsQueueFamily;
    barriers[1].dstQueueFamilyIndex = r.asyncComputeQueueFamily;
    barriers[1].buffer = r.indirectCommandBuffer;
    barriers[1].offset = 0;
    barriers[1].size = sizeof(TraceRaysIndirectCommandKHR);

    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.bufferMemoryBarrierCount = 2;
    dependency.pBufferMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(cmd, &dependency);
}

void insertCudaToGraphicsAcquireBarrier(VkCommandBuffer cmd, const FrameInteropResources& r) {
    VkBufferMemoryBarrier2 barriers[3]{};

    // Acquire CUDA-written TensorRT logits for optional Vulkan debugging or
    // shader-side inspection. Skip this barrier if logits never return to Vulkan.
    barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    barriers[0].srcAccessMask = VK_ACCESS_2_NONE;
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    barriers[0].srcQueueFamilyIndex = r.asyncComputeQueueFamily;
    barriers[0].dstQueueFamilyIndex = r.graphicsQueueFamily;
    barriers[0].buffer = r.logitsBuffer;
    barriers[0].offset = 0;
    barriers[0].size = VK_WHOLE_SIZE;

    // Tile budget map: CUDA writes compact per-tile ray counts, raygen reads it
    // as a storage buffer.
    barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    barriers[1].srcAccessMask = VK_ACCESS_2_NONE;
    barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    barriers[1].srcQueueFamilyIndex = r.asyncComputeQueueFamily;
    barriers[1].dstQueueFamilyIndex = r.graphicsQueueFamily;
    barriers[1].buffer = r.tileBudgetBuffer;
    barriers[1].offset = 0;
    barriers[1].size = VK_WHOLE_SIZE;

    // Indirect command: CUDA writes VkTraceRaysIndirectCommandKHR, the indirect
    // command processor reads it. Use INDIRECT_COMMAND_READ, not shader read.
    barriers[2].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barriers[2].srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    barriers[2].srcAccessMask = VK_ACCESS_2_NONE;
    barriers[2].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    barriers[2].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    barriers[2].srcQueueFamilyIndex = r.asyncComputeQueueFamily;
    barriers[2].dstQueueFamilyIndex = r.graphicsQueueFamily;
    barriers[2].buffer = r.indirectCommandBuffer;
    barriers[2].offset = 0;
    barriers[2].size = sizeof(TraceRaysIndirectCommandKHR);

    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.bufferMemoryBarrierCount = 3;
    dependency.pBufferMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(cmd, &dependency);
}

void runFrameIntegration(
    const FrameInteropResources& r,
    RayBudgetInferenceEngine& inferenceEngine,
    RayBudgetProfileHarness& profiler,
    PFN_vkCmdTraceRaysIndirectKHRChecked vkCmdTraceRaysIndirectKHR_,
    const VulkanTimestampContext* timestampCtx = nullptr) {
    if (r.cudaStream == nullptr) {
        throw std::runtime_error("FrameInteropResources.cudaStream is null");
    }
    if (vkCmdTraceRaysIndirectKHR_ == nullptr) {
        throw std::runtime_error("vkCmdTraceRaysIndirectKHR function pointer is null");
    }
    (void)inferenceEngine;  // The profiler owns the timed inference wrapper.

    /*
        1. Record and submit G-buffer pass on graphics queue.

        The G-buffer pass writes normal/depth/albedo into a tightly packed
        storage buffer matching [1,7,1080,1920]. This avoids tiled-image CUDA
        interpretation hazards.
    */
    profiler.recordVulkanGBufferExportBegin();
    insertGraphicsToCudaReleaseBarrier(r.graphicsCmd, r);
    profiler.recordVulkanGBufferExportEnd();

    /*
        2. Submit graphics command buffer and signal an external semaphore.

        Vulkan buffer barriers make writes available within Vulkan. CUDA cannot
        observe that barrier until an external semaphore is signaled by the queue
        and waited by the CUDA stream:

            vkQueueSubmit2(graphicsQueue, signal external timeline semaphore)
            cudaWaitExternalSemaphoresAsync(..., r.cudaStream)

        The semaphore wait is intentionally not hidden here; it belongs to the
        engine's frame graph and timeline ownership policy.
    */
    profiler.recordCudaInteropSyncBegin();
    profiler.recordCudaInteropSyncEnd();

    /*
        3. CUDA/TensorRT work on async stream.

        forwardAsync reads dGBufferTensor directly from the Vulkan allocation and
        writes dLogits. The converter then writes the tile budget map and the
        VkTraceRaysIndirectCommandKHR bytes into Vulkan-owned external buffers.
    */
    ProfileFrameInputs profileInputs{};
    profileInputs.dGBufferTensor = r.dGBufferTensor;
    profileInputs.dLogits = r.dLogits;
    profileInputs.dIndirectCommand = r.dIndirectCommand;
    profileInputs.dTileBudgetMap = r.dTileBudgetMap;
    profileInputs.dTotalRayCounter = r.dTotalRayCounter;
    profileInputs.cudaStream = r.cudaStream;
    profiler.runInferenceAndConverter(profileInputs);

    /*
        4. Signal a CUDA external semaphore, then have Vulkan wait before the
           acquire barrier and indirect ray dispatch:

            cudaSignalExternalSemaphoresAsync(..., r.cudaStream)
            vkQueueSubmit2(graphicsQueue, wait external timeline semaphore)

        The acquire barrier below is recorded in the command buffer that runs
        after the semaphore wait.
    */
    insertCudaToGraphicsAcquireBarrier(r.graphicsCmd, r);

    /*
        5. Ray-tracing indirect dispatch.

        indirectCommandDeviceAddress must point at a buffer containing exactly:
            uint32_t width;
            uint32_t height;
            uint32_t depth;

        The SBT regions are unchanged from the normal ray tracing path.
    */
    profiler.recordVulkanIndirectRayTraceBegin();
    if (timestampCtx != nullptr) {
        profiler.resetTimestampQueriesForFrame(*timestampCtx);
        profiler.writeRayTraceTimestampBegin(*timestampCtx);
    }
    vkCmdTraceRaysIndirectKHR_(
        r.graphicsCmd,
        &r.raygenSbt,
        &r.missSbt,
        &r.hitSbt,
        &r.callableSbt,
        r.indirectCommandDeviceAddress);
    if (timestampCtx != nullptr) {
        profiler.writeRayTraceTimestampEnd(*timestampCtx);
    }
    profiler.recordVulkanIndirectRayTraceEnd();
}

}  // namespace raybudget
