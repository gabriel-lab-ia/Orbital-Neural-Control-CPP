#include "interfaces/cli/command_line.h"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        return nmc::interfaces::cli::run_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
    }

    return 1;
}
