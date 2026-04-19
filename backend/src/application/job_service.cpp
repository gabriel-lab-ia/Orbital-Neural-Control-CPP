#include "application/job_service.h"

#include "common/json.h"
#include "common/logger.h"
#include "common/run_id.h"
#include "common/time_utils.h"

#include <cstdlib>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sstream>
#include <vector>

namespace orbital::backend::application {
namespace {

std::vector<std::string> command_args_for(const std::filesystem::path& repo_root, const JobLaunchRequest& request) {
    std::vector<std::string> args;
    args.reserve(16);
    args.push_back((repo_root / "build" / "nmc").string());
    args.push_back(domain::to_string(request.type));

    switch (request.type) {
        case domain::JobType::Train:
            args.push_back("--env");
            args.push_back("point_mass");
            args.push_back("--seed");
            args.push_back(std::to_string(request.seed));
            args.push_back("--run-id");
            args.push_back(request.run_id);
            if (request.quick) {
                args.push_back("--quick");
            }
            break;
        case domain::JobType::Eval:
            args.push_back("--env");
            args.push_back("point_mass");
            args.push_back("--seed");
            args.push_back(std::to_string(request.seed));
            args.push_back("--checkpoint");
            args.push_back("artifacts/latest/checkpoint.pt");
            args.push_back("--run-id");
            args.push_back(request.run_id);
            break;
        case domain::JobType::Benchmark:
            args.push_back("--seed");
            args.push_back(std::to_string(request.seed));
            args.push_back("--name");
            args.push_back(request.run_id);
            if (request.quick) {
                args.push_back("--quick");
            } else {
                args.push_back("--full");
            }
            break;
    }

    return args;
}

std::string command_debug_string(const std::vector<std::string>& args) {
    std::ostringstream stream;
    for (std::size_t index = 0; index < args.size(); ++index) {
        if (index > 0) {
            stream << ' ';
        }
        stream << args[index];
    }
    return stream.str();
}

int spawn_and_wait(const std::filesystem::path& working_directory, const std::vector<std::string>& args) {
    if (args.empty()) {
        return -1;
    }

    std::vector<char*> argv;
    argv.reserve(args.size() + 1U);
    for (const auto& argument : args) {
        argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        if (chdir(working_directory.string().c_str()) != 0) {
            _exit(126);
        }
        execv(argv[0], argv.data());
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;
}

std::string default_run_id(const domain::JobType type) {
    return std::string(domain::to_string(type)) + '_' + common::make_id("jobrun");
}

}  // namespace

JobService::JobService(std::filesystem::path repository_root, const bool executor_enabled)
    : repository_root_(std::move(repository_root)), executor_enabled_(executor_enabled) {}

JobService::~JobService() {
    workers_.clear();
}

domain::JobRecord JobService::submit(const JobLaunchRequest& request) {
    if (!request.run_id.empty()) {
        common::validate_run_id_or_throw(request.run_id, "run_id");
    }

    domain::JobRecord record;
    record.job_id = common::make_id("job");
    record.job_type = request.type;
    record.status = domain::JobStatus::Queued;
    record.run_id = request.run_id.empty() ? default_run_id(request.type) : request.run_id;
    common::validate_run_id_or_throw(record.run_id, "run_id");
    record.created_at = common::now_utc_iso8601();
    record.updated_at = record.created_at;

    {
        std::scoped_lock lock(mutex_);
        jobs_.insert_or_assign(record.job_id, record);
    }

    JobLaunchRequest launch_request = request;
    launch_request.run_id = record.run_id;

    workers_.emplace_back([this, job_id = record.job_id, launch_request](std::stop_token) mutable {
        run_job(job_id, std::move(launch_request));
    });

    return record;
}

std::optional<domain::JobRecord> JobService::get(const std::string& job_id) const {
    std::scoped_lock lock(mutex_);
    const auto it = jobs_.find(job_id);
    if (it == jobs_.end()) {
        return std::nullopt;
    }
    return it->second;
}

void JobService::run_job(std::string job_id, JobLaunchRequest request) {
    {
        std::scoped_lock lock(mutex_);
        auto it = jobs_.find(job_id);
        if (it == jobs_.end()) {
            return;
        }
        it->second.status = domain::JobStatus::Running;
        it->second.updated_at = common::now_utc_iso8601();
        it->second.details_json =
            "{\"executor_enabled\":" + std::string(executor_enabled_ ? "true" : "false") +
            ",\"mode\":\"spawn_execv\"}";
    }

    if (!executor_enabled_) {
        std::scoped_lock lock(mutex_);
        auto it = jobs_.find(job_id);
        if (it == jobs_.end()) {
            return;
        }
        it->second.status = domain::JobStatus::Completed;
        it->second.updated_at = common::now_utc_iso8601();
        it->second.details_json =
            "{\"mode\":\"dry_run\",\"message\":\"set ORBITAL_JOB_EXECUTOR=1 to execute nmc jobs from backend\"}";
        return;
    }

    common::validate_run_id_or_throw(request.run_id, "run_id");
    const auto args = command_args_for(repository_root_, request);
    const auto command = command_debug_string(args);
    common::log(common::LogLevel::Info, "executing job command: " + command, request.run_id);

    const int exit_code = spawn_and_wait(repository_root_, args);

    std::scoped_lock lock(mutex_);
    auto it = jobs_.find(job_id);
    if (it == jobs_.end()) {
        return;
    }

    it->second.updated_at = common::now_utc_iso8601();
    if (exit_code == 0) {
        it->second.status = domain::JobStatus::Completed;
        it->second.details_json =
            "{\"exit_code\":0,\"command\":" + common::json_string(command) + "}";
    } else {
        it->second.status = domain::JobStatus::Failed;
        it->second.details_json =
            "{\"exit_code\":" + std::to_string(exit_code) + ",\"command\":" + common::json_string(command) + "}";
    }
}

}  // namespace orbital::backend::application
