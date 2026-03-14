#include "app/training_app.h"

#include <c10/util/Exception.h>

#include <exception>
#include <iostream>

int main() {
    try {
        return nmc::run_training_app();
    } catch (const c10::Error& error) {
        std::cerr << "LibTorch error: " << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
    }

    return 1;
}
