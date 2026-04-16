#include "infrastructure/artifacts/run_manifest.h"

#include <sstream>

#include "common/json_utils.h"

namespace nmc::infrastructure::artifacts {

std::string render_run_manifest_json(const RunManifest& manifest) {
    std::ostringstream stream;
    stream << "{\n";
    stream << "  \"run_id\": \"" << common::json_escape(manifest.run_id) << "\",\n";
    stream << "  \"mode\": \"" << common::json_escape(manifest.mode) << "\",\n";
    stream << "  \"environment\": \"" << common::json_escape(manifest.environment) << "\",\n";
    stream << "  \"started_at\": \"" << common::json_escape(manifest.started_at) << "\",\n";
    stream << "  \"ended_at\": \"" << common::json_escape(manifest.ended_at) << "\",\n";
    stream << "  \"status\": \"" << common::json_escape(manifest.status) << "\",\n";
    if (manifest.checkpoint_path.has_value()) {
        stream << "  \"checkpoint\": \"" << common::json_escape(*manifest.checkpoint_path) << "\",\n";
    }
    if (manifest.error_message.has_value()) {
        stream << "  \"error\": \"" << common::json_escape(*manifest.error_message) << "\",\n";
    }
    stream << "  \"config\": " << manifest.config_json << ",\n";
    stream << "  \"artifacts\": {\n";
    for (std::size_t index = 0; index < manifest.artifacts.size(); ++index) {
        const auto& artifact = manifest.artifacts[index];
        stream << "    \"" << common::json_escape(artifact.name) << "\": \""
               << common::json_escape(artifact.path.string()) << "\"";
        if (index + 1 < manifest.artifacts.size()) {
            stream << ',';
        }
        stream << '\n';
    }
    stream << "  }\n";
    stream << "}\n";
    return stream.str();
}

}  // namespace nmc::infrastructure::artifacts
