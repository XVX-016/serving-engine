#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <sstream>

namespace engine {

// Minimal NPY writer supporting float (float32) and int64_t (int64)
class NpyWriter {
public:
    static void write_f32(const std::string& path, const float* data, const std::vector<int64_t>& shape) {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Failed to open file for writing NPY: " + path);
        }

        // Magic string "\x93NUMPY"
        const char magic[] = "\x93NUMPY";
        f.write(magic, 6);

        // Major and minor version: 1.0
        const char version[2] = {1, 0};
        f.write(version, 2);

        // Header string dictionary
        std::ostringstream dict;
        dict << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
        for (size_t i = 0; i < shape.size(); ++i) {
            dict << shape[i];
            if (shape.size() == 1 || i + 1 < shape.size()) {
                dict << ", ";
            }
        }
        dict << ")}";

        std::string dict_str = dict.str();
        // Total header length: 10 bytes prelude (6 magic + 2 version + 2 len) + dict_str.size() + padding + 1 newline
        // Total header (prelude + header) must be divisible by 64.
        // Therefore (10 + dict_str.size() + padding + 1) % 64 == 0.
        // Header_len is (dict_str.size() + padding + 1)
        size_t rem = (10 + dict_str.size() + 1) % 64;
        size_t pad_len = (rem == 0) ? 0 : (64 - rem);
        std::string padding(pad_len, ' ');

        std::string full_header = dict_str + padding + "\n";
        uint16_t header_len = static_cast<uint16_t>(full_header.size());

        // Write header len (little-endian)
        f.write(reinterpret_cast<const char*>(&header_len), 2);
        // Write header
        f.write(full_header.data(), full_header.size());

        // Write raw float32 data
        size_t total_elements = 1;
        for (int64_t dim : shape) {
            total_elements *= static_cast<size_t>(dim);
        }
        f.write(reinterpret_cast<const char*>(data), total_elements * sizeof(float));
    }

    static void write_i64(const std::string& path, const int64_t* data, const std::vector<int64_t>& shape) {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Failed to open file for writing NPY: " + path);
        }

        const char magic[] = "\x93NUMPY";
        f.write(magic, 6);

        const char version[2] = {1, 0};
        f.write(version, 2);

        std::ostringstream dict;
        dict << "{'descr': '<i8', 'fortran_order': False, 'shape': (";
        for (size_t i = 0; i < shape.size(); ++i) {
            dict << shape[i];
            if (shape.size() == 1 || i + 1 < shape.size()) {
                dict << ", ";
            }
        }
        dict << ")}";

        std::string dict_str = dict.str();
        size_t rem = (10 + dict_str.size() + 1) % 64;
        size_t pad_len = (rem == 0) ? 0 : (64 - rem);
        std::string padding(pad_len, ' ');

        std::string full_header = dict_str + padding + "\n";
        uint16_t header_len = static_cast<uint16_t>(full_header.size());

        f.write(reinterpret_cast<const char*>(&header_len), 2);
        f.write(full_header.data(), full_header.size());

        size_t total_elements = 1;
        for (int64_t dim : shape) {
            total_elements *= static_cast<size_t>(dim);
        }
        f.write(reinterpret_cast<const char*>(data), total_elements * sizeof(int64_t));
    }
};

} // namespace engine

