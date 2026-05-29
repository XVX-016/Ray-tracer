/*
    Vulkan <-> CUDA external memory bridge for Ray Budget Allocation.

    This file creates a device-local Vulkan buffer backed by exportable memory
    and imports the same allocation into CUDA. The returned CUDA device pointer
    is suitable for TensorRT input:

        input_gbuffers [1, 7, 1080, 1920] float32

    Important memory model note:
    TensorRT consumes a tightly packed linear tensor. An optimal-tiled VkImage
    cannot be safely treated as a linear CUDA pointer. The production path is:

        Vulkan G-buffer images -> Vulkan compute/graphics packing pass ->
        external VkBuffer -> CUDA/TensorRT

    The external memory object is therefore a VkBuffer allocation. Use a
    dedicated allocation with offset 0 for CUDA import robustness.
*/

#include <vulkan/vulkan.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace raybudget {

constexpr uint32_t kInputChannels = 7;
constexpr uint32_t kInputHeight = 1080;
constexpr uint32_t kInputWidth = 1920;
constexpr VkDeviceSize kGBufferElementCount =
    static_cast<VkDeviceSize>(kInputChannels) * kInputHeight * kInputWidth;
constexpr VkDeviceSize kGBufferBytes = kGBufferElementCount * sizeof(float);

#if defined(_WIN32)
constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

#define RB_CHECK_VK(expr) checkVk((expr), #expr, __FILE__, __LINE__)
#define RB_CHECK_CUDA(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

void checkVk(VkResult result, const char* expr, const char* file, int line) {
    if (result != VK_SUCCESS) {
        std::ostringstream oss;
        oss << "Vulkan call failed: " << expr << " at " << file << ":" << line
            << " VkResult=" << static_cast<int>(result);
        throw std::runtime_error(oss.str());
    }
}

void checkCuda(cudaError_t result, const char* expr, const char* file, int line) {
    if (result != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA call failed: " << expr << " at " << file << ":" << line
            << " " << cudaGetErrorString(result)
            << " (" << static_cast<int>(result) << ")";
        throw std::runtime_error(oss.str());
    }
}

uint32_t findMemoryType(
    VkPhysicalDevice physicalDevice,
    uint32_t memoryTypeBits,
    VkMemoryPropertyFlags requiredFlags) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &props);

    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        const bool supported = (memoryTypeBits & (1u << i)) != 0;
        const bool hasFlags =
            (props.memoryTypes[i].propertyFlags & requiredFlags) == requiredFlags;
        if (supported && hasFlags) {
            return i;
        }
    }
    throw std::runtime_error("No compatible Vulkan memory type found");
}

PFN_vkGetMemoryFdKHR loadGetMemoryFd(VkDevice device) {
    auto fn = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
    if (fn == nullptr) {
        throw std::runtime_error("vkGetMemoryFdKHR not loaded");
    }
    return fn;
}

#if defined(_WIN32)
PFN_vkGetMemoryWin32HandleKHR loadGetMemoryWin32Handle(VkDevice device) {
    auto fn = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
    if (fn == nullptr) {
        throw std::runtime_error("vkGetMemoryWin32HandleKHR not loaded");
    }
    return fn;
}
#endif

struct ExternalGBufferTensor {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer vkBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vkMemory = VK_NULL_HANDLE;
    VkDeviceSize byteSize = kGBufferBytes;

    cudaExternalMemory_t cudaExternalMemory = nullptr;
    void* cudaDevicePtr = nullptr;

#if defined(_WIN32)
    HANDLE osHandle = nullptr;
#else
    int osFd = -1;
#endif

    void destroy() noexcept {
        if (cudaDevicePtr != nullptr) {
            cudaFree(cudaDevicePtr);
            cudaDevicePtr = nullptr;
        }
        if (cudaExternalMemory != nullptr) {
            cudaDestroyExternalMemory(cudaExternalMemory);
            cudaExternalMemory = nullptr;
        }
#if defined(_WIN32)
        if (osHandle != nullptr) {
            CloseHandle(osHandle);
            osHandle = nullptr;
        }
#else
        if (osFd >= 0) {
            close(osFd);
            osFd = -1;
        }
#endif
        if (device != VK_NULL_HANDLE && vkBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, vkBuffer, nullptr);
            vkBuffer = VK_NULL_HANDLE;
        }
        if (device != VK_NULL_HANDLE && vkMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, vkMemory, nullptr);
            vkMemory = VK_NULL_HANDLE;
        }
        device = VK_NULL_HANDLE;
    }
};

ExternalGBufferTensor createExternalGBufferTensor(
    VkPhysicalDevice physicalDevice,
    VkDevice device) {
    ExternalGBufferTensor tensor{};
    tensor.device = device;

    VkExternalMemoryBufferCreateInfo externalInfo{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    externalInfo.handleTypes = kExternalMemoryHandleType;

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.pNext = &externalInfo;
    bufferInfo.size = kGBufferBytes;
    bufferInfo.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    RB_CHECK_VK(vkCreateBuffer(device, &bufferInfo, nullptr, &tensor.vkBuffer));

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, tensor.vkBuffer, &requirements);
    if (requirements.size < kGBufferBytes) {
        throw std::runtime_error("Vulkan memory requirements smaller than tensor byte size");
    }

    const uint32_t memoryTypeIndex = findMemoryType(
        physicalDevice,
        requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkExportMemoryAllocateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportInfo.pNext = nullptr;
    exportInfo.handleTypes = kExternalMemoryHandleType;

    VkMemoryDedicatedAllocateInfo dedicatedInfo{
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicatedInfo.pNext = &exportInfo;
    dedicatedInfo.image = VK_NULL_HANDLE;
    dedicatedInfo.buffer = tensor.vkBuffer;

    VkMemoryAllocateFlagsInfo flagsInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flagsInfo.pNext = &dedicatedInfo;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    flagsInfo.deviceMask = 0;

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.pNext = &flagsInfo;
    allocInfo.allocationSize = requirements.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    RB_CHECK_VK(vkAllocateMemory(device, &allocInfo, nullptr, &tensor.vkMemory));
    RB_CHECK_VK(vkBindBufferMemory(device, tensor.vkBuffer, tensor.vkMemory, 0));

#if defined(_WIN32)
    VkMemoryGetWin32HandleInfoKHR handleInfo{
        VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    handleInfo.memory = tensor.vkMemory;
    handleInfo.handleType = kExternalMemoryHandleType;
    RB_CHECK_VK(loadGetMemoryWin32Handle(device)(device, &handleInfo, &tensor.osHandle));

    cudaExternalMemoryHandleDesc cudaDesc{};
    cudaDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    cudaDesc.handle.win32.handle = tensor.osHandle;
    cudaDesc.size = static_cast<unsigned long long>(requirements.size);
#else
    VkMemoryGetFdInfoKHR fdInfo{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    fdInfo.memory = tensor.vkMemory;
    fdInfo.handleType = kExternalMemoryHandleType;
    RB_CHECK_VK(loadGetMemoryFd(device)(device, &fdInfo, &tensor.osFd));

    cudaExternalMemoryHandleDesc cudaDesc{};
    cudaDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    cudaDesc.handle.fd = tensor.osFd;
    cudaDesc.size = static_cast<unsigned long long>(requirements.size);
#endif

    RB_CHECK_CUDA(cudaImportExternalMemory(&tensor.cudaExternalMemory, &cudaDesc));

    cudaExternalMemoryBufferDesc bufferDesc{};
    bufferDesc.offset = 0;
    bufferDesc.size = static_cast<unsigned long long>(kGBufferBytes);
    bufferDesc.flags = 0;
    RB_CHECK_CUDA(cudaExternalMemoryGetMappedBuffer(
        &tensor.cudaDevicePtr,
        tensor.cudaExternalMemory,
        &bufferDesc));

#if !defined(_WIN32)
    // CUDA duplicates/imports the FD; the original Vulkan-exported FD should
    // not remain open after import in this process.
    close(tensor.osFd);
    tensor.osFd = -1;
#endif

    return tensor;
}

VkDeviceAddress getExternalGBufferDeviceAddress(
    VkDevice device,
    VkBuffer buffer) {
    VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addressInfo.buffer = buffer;
    const VkDeviceAddress address = vkGetBufferDeviceAddress(device, &addressInfo);
    if (address == 0) {
        throw std::runtime_error("vkGetBufferDeviceAddress returned 0");
    }
    return address;
}

}  // namespace raybudget
