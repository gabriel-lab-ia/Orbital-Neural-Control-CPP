#include "domain/inference/inference_backend_factory.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

#include "domain/inference/libtorch_policy_backend.h"
#include "domain/inference/tensorrt_policy_backend_stub.h"

namespace nmc::domain::inference {

std::optional<InferenceBackendKind> try_parse_inference_backend(const std::string_view backend) {
    std::string normalized(backend);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (normalized.empty() || normalized == "libtorch") {
        return InferenceBackendKind::kLibTorch;
    }

    if (normalized == "tensorrt" || normalized == "tensorrt_stub") {
        return InferenceBackendKind::kTensorRtStub;
    }

    return std::nullopt;
}

InferenceBackendKind parse_inference_backend_or_throw(const std::string_view backend) {
    const auto parsed = try_parse_inference_backend(backend);
    if (!parsed.has_value()) {
        throw std::runtime_error(
            "unsupported inference backend: " + std::string(backend) +
            " (supported: " + supported_inference_backends() + ")"
        );
    }
    return *parsed;
}

std::string inference_backend_to_string(const InferenceBackendKind backend) {
    switch (backend) {
        case InferenceBackendKind::kLibTorch:
            return "libtorch";
        case InferenceBackendKind::kTensorRtStub:
            return "tensorrt_stub";
    }
    return "unknown";
}

std::string supported_inference_backends() {
    return "libtorch|tensorrt_stub";
}

std::unique_ptr<PolicyInferenceBackend> make_inference_backend(
    const InferenceBackendKind backend,
    const int64_t observation_dim,
    const int64_t action_dim,
    const int64_t hidden_dim
) {
    switch (backend) {
        case InferenceBackendKind::kLibTorch:
            return std::make_unique<LibTorchPolicyBackend>(observation_dim, action_dim, hidden_dim);
        case InferenceBackendKind::kTensorRtStub:
            return std::make_unique<TensorRtPolicyBackendStub>();
    }
    throw std::runtime_error("unsupported inference backend kind");
}

}  // namespace nmc::domain::inference
