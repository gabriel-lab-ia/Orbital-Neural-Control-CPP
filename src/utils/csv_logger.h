#pragma once

#include <filesystem>
#include <fstream>
#include <string>

#include "train/rollout_buffer.h"

namespace nmc {

class CsvLogger {
public:
    explicit CsvLogger(const std::filesystem::path& path);
    ~CsvLogger();

    void log(const TrainingMetrics& metrics);

private:
    std::ofstream stream_;
};

}  // namespace nmc
