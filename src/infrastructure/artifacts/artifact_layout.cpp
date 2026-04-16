#include "infrastructure/artifacts/artifact_layout.h"

#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace nmc::infrastructure::artifacts {
namespace {

bool is_same_path(const std::filesystem::path& lhs, const std::filesystem::path& rhs) {
    std::error_code ec_lhs;
    std::error_code ec_rhs;
    const auto lhs_abs = std::filesystem::absolute(lhs, ec_lhs);
    const auto rhs_abs = std::filesystem::absolute(rhs, ec_rhs);
    if (ec_lhs || ec_rhs) {
        return false;
    }
    return lhs_abs.lexically_normal() == rhs_abs.lexically_normal();
}

std::filesystem::path make_temp_path_for(const std::filesystem::path& destination) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::ostringstream stream;
    stream << destination.filename().string() << ".tmp." << now;
    return destination.parent_path() / stream.str();
}

}  // namespace

ArtifactLayout make_layout(const std::filesystem::path& root, const std::string& run_id) {
    ArtifactLayout layout;
    layout.root = root;
    layout.runs_dir = root / "runs";
    layout.checkpoints_dir = root / "checkpoints";
    layout.reports_dir = root / "reports";
    layout.benchmarks_dir = root / "benchmarks";
    layout.latest_dir = root / "latest";

    layout.run_dir = layout.runs_dir / run_id;
    layout.run_manifest_json = layout.run_dir / "manifest.json";
    layout.train_metrics_csv = layout.run_dir / "training_metrics.csv";
    layout.train_summary_json = layout.run_dir / "training_summary.json";
    layout.eval_summary_json = layout.run_dir / "evaluation_summary.json";
    layout.live_rollout_csv = layout.run_dir / "live_rollout.csv";

    layout.run_checkpoint_dir = layout.run_dir / "checkpoints";
    layout.run_checkpoint_model = layout.run_checkpoint_dir / "policy_last.pt";
    layout.run_checkpoint_meta = layout.run_checkpoint_dir / "policy_last.meta";

    layout.global_checkpoint_model = layout.checkpoints_dir / (run_id + "_policy_last.pt");
    layout.global_checkpoint_meta = layout.checkpoints_dir / (run_id + "_policy_last.meta");

    std::filesystem::create_directories(layout.runs_dir);
    std::filesystem::create_directories(layout.checkpoints_dir);
    std::filesystem::create_directories(layout.reports_dir);
    std::filesystem::create_directories(layout.benchmarks_dir);
    std::filesystem::create_directories(layout.latest_dir);
    std::filesystem::create_directories(layout.run_dir);
    std::filesystem::create_directories(layout.run_checkpoint_dir);

    return layout;
}

void write_text_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    const auto temp_path = make_temp_path_for(path);
    std::ofstream stream(temp_path, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("unable to open file: " + temp_path.string());
    }
    stream << content;
    if (!stream.good()) {
        throw std::runtime_error("failed to write file: " + temp_path.string());
    }
    stream.flush();
    stream.close();

    std::error_code rename_error;
    std::filesystem::rename(temp_path, path, rename_error);
    if (!rename_error) {
        return;
    }

    std::error_code remove_error;
    std::filesystem::remove(path, remove_error);
    rename_error.clear();
    std::filesystem::rename(temp_path, path, rename_error);
    if (rename_error) {
        std::error_code cleanup_error;
        std::filesystem::remove(temp_path, cleanup_error);
        throw std::runtime_error(
            "failed to atomically replace file: " + path.string() + " (" + rename_error.message() + ")"
        );
    }
}

bool is_readable_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::in | std::ios::binary);
    return stream.good();
}

void refresh_latest_snapshot(
    const ArtifactLayout& layout,
    const std::vector<std::filesystem::path>& files_to_copy,
    const std::filesystem::path& checkpoint_model,
    const std::filesystem::path& checkpoint_meta
) {
    std::filesystem::create_directories(layout.latest_dir);

    for (const auto& file : files_to_copy) {
        if (std::filesystem::exists(file)) {
            const auto destination = layout.latest_dir / file.filename();
            if (!is_same_path(file, destination)) {
                std::filesystem::copy_file(file, destination, std::filesystem::copy_options::overwrite_existing);
            }
        }
    }

    if (std::filesystem::exists(checkpoint_model)) {
        const auto destination = layout.latest_dir / "checkpoint.pt";
        if (!is_same_path(checkpoint_model, destination)) {
            std::filesystem::copy_file(
                checkpoint_model,
                destination,
                std::filesystem::copy_options::overwrite_existing
            );
        }
    }

    if (std::filesystem::exists(checkpoint_meta)) {
        const auto destination = layout.latest_dir / "checkpoint.meta";
        if (!is_same_path(checkpoint_meta, destination)) {
            std::filesystem::copy_file(
                checkpoint_meta,
                destination,
                std::filesystem::copy_options::overwrite_existing
            );
        }
    }
}

}  // namespace nmc::infrastructure::artifacts
