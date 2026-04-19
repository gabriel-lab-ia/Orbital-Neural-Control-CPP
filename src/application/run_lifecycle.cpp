#include "application/run_lifecycle.h"

#include "common/json_utils.h"

namespace nmc::application {
namespace {

std::string failed_summary_json(const std::string& run_id, const std::string& error_message) {
    return "{\"status\":\"failed\",\"run_id\":\"" +
        common::json_escape(run_id) +
        "\",\"error\":\"" + common::json_escape(error_message) + "\"}";
}

}  // namespace

void record_run_start(
    infrastructure::persistence::SQLiteExperimentStore& db,
    const RunStartContext& context
) {
    db.insert_run_start(
        {
            context.run_id,
            context.mode,
            context.environment,
            context.seed,
            context.started_at,
            "running",
            context.artifact_dir.string(),
            context.config_json
        }
    );
}

void record_run_success(
    infrastructure::persistence::SQLiteExperimentStore& db,
    const std::string& run_id,
    const std::string& ended_at,
    const std::string& summary_json
) {
    db.finalize_run(run_id, "completed", ended_at, summary_json);
}

void record_run_failure(
    infrastructure::persistence::SQLiteExperimentStore& db,
    const infrastructure::artifacts::ArtifactLayout& layout,
    const RunFailureContext& context
) {
    const auto failed_manifest = infrastructure::artifacts::render_run_manifest_json(
        {
            context.run_id,
            context.mode,
            context.environment,
            context.started_at,
            context.failed_at,
            "failed",
            context.config_json,
            context.artifacts,
            context.checkpoint_path,
            context.error_message
        }
    );
    infrastructure::artifacts::write_text_file(layout.run_manifest_json, failed_manifest);

    db.insert_event(
        {
            context.run_id,
            "error",
            "run_failed",
            context.error_message,
            "{}",
            context.failed_at
        }
    );
    db.finalize_run(
        context.run_id,
        "failed",
        context.failed_at,
        failed_summary_json(context.run_id, context.error_message)
    );
}

}  // namespace nmc::application
