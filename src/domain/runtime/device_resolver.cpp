#include "domain/runtime/device_resolver.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

#include <torch/cuda.h>

#include "common/json_utils.h"

namespace nmc::domain::runtime {
namespace {

std::string lower_copy(const std::string_view value) {
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return normalized;
}

}  // namespace

ComputeBackend parse_compute_backend_or_throw(const std::string_view value) {
    const auto normalized = lower_copy(value);
    if (normalized == "cpu") {
        return ComputeBackend::kCpu;
    }
    if (normalized == "cuda") {
        return ComputeBackend::kCuda;
    }
    if (normalized == "auto") {
        return ComputeBackend::kAuto;
    }
    throw std::runtime_error(
        "unsupported compute device: " + std::string(value) +
        " (supported: " + supported_compute_backends() + ")"
    );
}

std::string compute_backend_to_string(const ComputeBackend backend) {
    switch (backend) {
        case ComputeBackend::kCpu:
            return "cpu";
        case ComputeBackend::kCuda:
            return "cuda";
        case ComputeBackend::kAuto:
            return "auto";
    }
    return "unknown";
}

std::string supported_compute_backends() {
    return "cpu|cuda|auto";
}

ResolvedDevice resolve_device(const DeviceConfig& config) {
    if (config.cuda_device_index < 0) {
        throw std::runtime_error("--cuda-device must be >= 0");
    }

    ResolvedDevice resolved;
    resolved.requested = config;
    resolved.cuda_available = torch::cuda::is_available();
    resolved.cuda_device_count = static_cast<int64_t>(torch::cuda::device_count());

    const bool valid_cuda_index =
        resolved.cuda_available && config.cuda_device_index < resolved.cuda_device_count;
    const bool wants_cuda =
        config.backend == ComputeBackend::kCuda || config.backend == ComputeBackend::kAuto;

    if (wants_cuda && valid_cuda_index) {
        resolved.resolved_backend = ComputeBackend::kCuda;
        resolved.torch_device = torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(config.cuda_device_index));
        // The portable LibTorch C++ API exposes availability/count without requiring CUDA SDK headers.
        resolved.cuda_device_name = "cuda:" + std::to_string(config.cuda_device_index);
        return resolved;
    }

    if (config.backend == ComputeBackend::kCuda && !config.allow_fallback) {
        std::ostringstream message;
        message << "CUDA device " << config.cuda_device_index << " was requested but is unavailable"
                << " (cuda_available=" << (resolved.cuda_available ? "true" : "false")
                << ", cuda_device_count=" << resolved.cuda_device_count << ')';
        throw std::runtime_error(message.str());
    }

    resolved.resolved_backend = ComputeBackend::kCpu;
    resolved.torch_device = torch::Device(torch::kCPU);
    resolved.cuda_fallback_used = wants_cuda;
    return resolved;
}

std::string device_metadata_json(const ResolvedDevice& device) {
    std::ostringstream stream;
    stream << '{';
    stream << "\"compute_backend_requested\":\""
           << common::json_escape(compute_backend_to_string(device.requested.backend)) << "\",";
    stream << "\"compute_backend_resolved\":\""
           << common::json_escape(compute_backend_to_string(device.resolved_backend)) << "\",";
    stream << "\"torch_device\":\"" << common::json_escape(device.torch_device.str()) << "\",";
    stream << "\"cuda_available\":" << (device.cuda_available ? "true" : "false") << ',';
    stream << "\"cuda_device_count\":" << device.cuda_device_count << ',';
    stream << "\"cuda_device_index\":" << device.requested.cuda_device_index << ',';
    stream << "\"cuda_device_name\":\"" << common::json_escape(device.cuda_device_name) << "\",";
    stream << "\"cuda_fallback_used\":" << (device.cuda_fallback_used ? "true" : "false");
    stream << '}';
    return stream.str();
}

}  // namespace nmc::domain::runtime
