#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include <torch/torch.h>

namespace nmc::domain::runtime {

enum class ComputeBackend {
    kCpu,
    kCuda,
    kAuto
};

struct DeviceConfig {
    ComputeBackend backend = ComputeBackend::kCpu;
    int64_t cuda_device_index = 0;
    bool allow_fallback = true;
};

struct ResolvedDevice {
    DeviceConfig requested{};
    ComputeBackend resolved_backend = ComputeBackend::kCpu;
    torch::Device torch_device{torch::kCPU};
    bool cuda_available = false;
    int64_t cuda_device_count = 0;
    std::string cuda_device_name;
    bool cuda_fallback_used = false;
};

ComputeBackend parse_compute_backend_or_throw(std::string_view value);
std::string compute_backend_to_string(ComputeBackend backend);
std::string supported_compute_backends();
ResolvedDevice resolve_device(const DeviceConfig& config);
std::string device_metadata_json(const ResolvedDevice& device);

}  // namespace nmc::domain::runtime
