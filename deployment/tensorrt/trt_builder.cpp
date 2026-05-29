#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kInputName[] = "input_gbuffers";
constexpr char kOutputName[] = "output_tile_logits";
constexpr int32_t kBatch = 1;
constexpr int32_t kInputC = 7;
constexpr int32_t kInputH = 1080;
constexpr int32_t kInputW = 1920;
constexpr int32_t kOutputC = 6;
constexpr int32_t kOutputTileX = 120;
constexpr int32_t kOutputTileY = 68;

#define CHECK_CUDA(call)                                                                 \
    do {                                                                                 \
        cudaError_t status__ = (call);                                                    \
        if (status__ != cudaSuccess) {                                                    \
            std::ostringstream oss__;                                                     \
            oss__ << "CUDA failure at " << __FILE__ << ":" << __LINE__ << " : "         \
                  << cudaGetErrorString(status__) << " (" << static_cast<int>(status__)   \
                  << ")";                                                                \
            throw std::runtime_error(oss__.str());                                        \
        }                                                                                \
    } while (0)

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cerr << "[TensorRT] " << severityToString(severity) << ": " << msg << '\n';
        }
    }

private:
    static const char* severityToString(Severity severity) noexcept {
        switch (severity) {
            case Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
            case Severity::kERROR: return "ERROR";
            case Severity::kWARNING: return "WARNING";
            case Severity::kINFO: return "INFO";
            case Severity::kVERBOSE: return "VERBOSE";
            default: return "UNKNOWN";
        }
    }
};

template <typename T>
struct TrtDestroy {
    void operator()(T* ptr) const noexcept {
        delete ptr;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroy<T>>;

std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open input file: " + path);
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("Input file is empty: " + path);
    }
    std::vector<char> bytes(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(bytes.data(), size)) {
        throw std::runtime_error("Failed to read file: " + path);
    }
    return bytes;
}

void writeFile(const std::string& path, const void* data, size_t size) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!file) {
        throw std::runtime_error("Failed to write output file: " + path);
    }
}

void printUsage(const char* exe) {
    std::cerr << "Usage:\n"
              << "  " << exe << " --onnx model.onnx --engine ray_budget_fp16.engine "
              << "[--workspace-mib 2048]\n";
}

struct Args {
    std::string onnxPath;
    std::string enginePath;
    size_t workspaceMiB = 2048;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto requireValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (key == "--onnx") {
            args.onnxPath = requireValue("--onnx");
        } else if (key == "--engine") {
            args.enginePath = requireValue("--engine");
        } else if (key == "--workspace-mib") {
            args.workspaceMiB = static_cast<size_t>(std::stoull(requireValue("--workspace-mib")));
        } else if (key == "--help" || key == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    if (args.onnxPath.empty() || args.enginePath.empty()) {
        printUsage(argv[0]);
        throw std::runtime_error("Both --onnx and --engine are required");
    }
    return args;
}

void setFp16LayerPrecision(nvinfer1::INetworkDefinition& network) {
    for (int32_t i = 0; i < network.getNbLayers(); ++i) {
        nvinfer1::ILayer* layer = network.getLayer(i);
        if (layer == nullptr) {
            continue;
        }
        layer->setPrecision(nvinfer1::DataType::kHALF);
        for (int32_t j = 0; j < layer->getNbOutputs(); ++j) {
            layer->setOutputType(j, nvinfer1::DataType::kHALF);
        }
    }

    nvinfer1::ITensor* output = network.getOutput(0);
    if (output == nullptr) {
        throw std::runtime_error("Network has no output tensor");
    }
    output->setType(nvinfer1::DataType::kFLOAT);
}

void validateNetworkIO(nvinfer1::INetworkDefinition& network) {
    if (network.getNbInputs() != 1) {
        throw std::runtime_error("Expected exactly one network input");
    }
    if (network.getNbOutputs() != 1) {
        throw std::runtime_error("Expected exactly one network output");
    }

    nvinfer1::ITensor* input = network.getInput(0);
    nvinfer1::ITensor* output = network.getOutput(0);
    if (std::string(input->getName()) != kInputName) {
        throw std::runtime_error(std::string("Unexpected input tensor name: ") + input->getName());
    }
    if (std::string(output->getName()) != kOutputName) {
        throw std::runtime_error(std::string("Unexpected output tensor name: ") + output->getName());
    }
}

}  // namespace

int main(int argc, char** argv) {
    Logger logger;
    try {
        const Args args = parseArgs(argc, argv);
        CHECK_CUDA(cudaFree(nullptr));

        TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        if (!builder) {
            throw std::runtime_error("createInferBuilder returned nullptr");
        }

        const uint32_t networkFlags =
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(networkFlags));
        if (!network) {
            throw std::runtime_error("createNetworkV2 returned nullptr");
        }

        TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
        if (!parser) {
            throw std::runtime_error("createParser returned nullptr");
        }

        const std::vector<char> onnx = readFile(args.onnxPath);
        if (!parser->parse(onnx.data(), onnx.size())) {
            std::ostringstream oss;
            oss << "ONNX parse failed with " << parser->getNbErrors() << " parser errors";
            for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
                const nvonnxparser::IParserError* error = parser->getError(i);
                if (error != nullptr) {
                    oss << "\n  [" << i << "] " << error->desc();
                }
            }
            throw std::runtime_error(oss.str());
        }

        validateNetworkIO(*network);

        TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (!config) {
            throw std::runtime_error("createBuilderConfig returned nullptr");
        }

        if (!builder->platformHasFastFp16()) {
            throw std::runtime_error("This platform does not report fast FP16 support");
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
        config->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE,
            args.workspaceMiB * 1024ULL * 1024ULL);
        setFp16LayerPrecision(*network);

        TrtUniquePtr<nvinfer1::IOptimizationProfile> profile(builder->createOptimizationProfile());
        if (!profile) {
            throw std::runtime_error("createOptimizationProfile returned nullptr");
        }
        const nvinfer1::Dims4 inputDims{kBatch, kInputC, kInputH, kInputW};
        if (!profile->setDimensions(kInputName, nvinfer1::OptProfileSelector::kMIN, inputDims) ||
            !profile->setDimensions(kInputName, nvinfer1::OptProfileSelector::kOPT, inputDims) ||
            !profile->setDimensions(kInputName, nvinfer1::OptProfileSelector::kMAX, inputDims)) {
            throw std::runtime_error("Failed to set optimization profile dimensions");
        }
        if (!profile->isValid()) {
            throw std::runtime_error("Optimization profile is invalid");
        }
        const int32_t profileIndex = config->addOptimizationProfile(profile.get());
        if (profileIndex < 0) {
            throw std::runtime_error("addOptimizationProfile failed");
        }

        std::cerr << "Building TensorRT FP16 engine from " << args.onnxPath << '\n';
        TrtUniquePtr<nvinfer1::IHostMemory> serialized(
            builder->buildSerializedNetwork(*network, *config));
        if (!serialized) {
            throw std::runtime_error("buildSerializedNetwork failed");
        }

        writeFile(args.enginePath, serialized->data(), serialized->size());
        std::cerr << "Engine written: " << args.enginePath
                  << " (" << serialized->size() << " bytes)\n"
                  << "Input : " << kInputName << " [1,7,1080,1920]\n"
                  << "Output: " << kOutputName << " [1,6,120,68]\n";
        return 0;
    } catch (const std::exception& ex) {
        logger.log(nvinfer1::ILogger::Severity::kERROR, ex.what());
        return 1;
    }
}
