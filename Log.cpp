#include "Log.h"

Log& Log::instance() {
    static Log instance;
    return instance;
}

void Log::open(std::string filename) {
    file_.open(filename, std::ios::out | std::ios::trunc);
}

void Log::debug(std::string message) {
    std::cout << message;
    file_ << message;   
}

void Log::close() {
    file_.close();
}