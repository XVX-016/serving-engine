#include "src/loader/safetensors.h"
#include "third_party/nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace engine {

int64_t TensorInfo::numel() const {
    if (shape.empty()) return 0;
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

size_t TensorInfo::nbytes() const {
    size_t elem_size = 0;
    switch (dtype) {
        case DataType::BF16:
        case DataType::F16: elem_size = 2; break;
        case DataType::F32: elem_size = 4; break;
        default: elem_size = 1; break;
    }
    return static_cast<size_t>(numel()) * elem_size;
}

#ifdef _WIN32
class WindowsFileMapping : public IFileMapping {
public:
    WindowsFileMapping(const std::string& path) {
        // Convert UTF-8 path to wide string
        int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        if (wlen <= 0) {
            throw std::runtime_error("Failed to convert path to UTF-16: " + path);
        }
        std::wstring wpath(wlen, 0);
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wpath[0], wlen);

        file_handle_ = CreateFileW(
            wpath.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Failed to open file: " + path + " (error: " + std::to_string(GetLastError()) + ")");
        }

        LARGE_INTEGER li_size;
        if (!GetFileSizeEx(file_handle_, &li_size)) {
            CloseHandle(file_handle_);
            throw std::runtime_error("Failed to get file size: " + path);
        }
        size_ = static_cast<size_t>(li_size.QuadPart);

        mapping_handle_ = CreateFileMappingW(
            file_handle_,
            nullptr,
            PAGE_READONLY,
            0,
            0,
            nullptr
        );
        if (!mapping_handle_) {
            CloseHandle(file_handle_);
            throw std::runtime_error("Failed to create file mapping: " + path + " (error: " + std::to_string(GetLastError()) + ")");
        }

        data_ = static_cast<const uint8_t*>(MapViewOfFile(
            mapping_handle_,
            FILE_MAP_READ,
            0,
            0,
            0
        ));
        if (!data_) {
            CloseHandle(mapping_handle_);
            CloseHandle(file_handle_);
            throw std::runtime_error("Failed to map view of file: " + path + " (error: " + std::to_string(GetLastError()) + ")");
        }
    }

    ~WindowsFileMapping() override {
        if (data_) UnmapViewOfFile(data_);
        if (mapping_handle_) CloseHandle(mapping_handle_);
        if (file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(file_handle_);
    }

    const uint8_t* data() const override { return data_; }
    size_t size() const override { return size_; }

private:
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
};
#else
class PosixFileMapping : public IFileMapping {
public:
    PosixFileMapping(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("Failed to open file: " + path);
        }
        struct stat sb;
        if (fstat(fd, &sb) < 0) {
            ::close(fd);
            throw std::runtime_error("Failed to fstat file: " + path);
        }
        size_ = sb.st_size;
        void* addr = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (addr == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap file: " + path);
        }
        data_ = static_cast<const uint8_t*>(addr);
    }

    ~PosixFileMapping() override {
        if (data_) munmap(const_cast<uint8_t*>(data_), size_);
    }

    const uint8_t* data() const override { return data_; }
    size_t size() const override { return size_; }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
};
#endif

bool SafetensorsLoader::open(const std::string& filepath) {
    close();
    try {
#ifdef _WIN32
        mapping_ = std::make_unique<WindowsFileMapping>(filepath);
#else
        mapping_ = std::make_unique<PosixFileMapping>(filepath);
#endif
    } catch (const std::exception& e) {
        std::cerr << "SafetensorsLoader open error: " << e.what() << std::endl;
        return false;
    }

    if (mapping_->size() < 8) {
        std::cerr << "Safetensors file too small (< 8 bytes)" << std::endl;
        close();
        return false;
    }

    const uint8_t* ptr = mapping_->data();
    uint64_t header_len = 0;
    std::memcpy(&header_len, ptr, 8);

    header_size_ = static_cast<size_t>(header_len);
    if (8 + header_size_ > mapping_->size()) {
        std::cerr << "Safetensors header size exceeds file length" << std::endl;
        close();
        return false;
    }

    tensor_buffer_ = ptr + 8 + header_size_;
    const size_t total_tensor_buffer_len = mapping_->size() - (8 + header_size_);

    try {
        std::string json_str(reinterpret_cast<const char*>(ptr + 8), header_size_);
        auto j = nlohmann::json::parse(json_str);

        for (auto it = j.begin(); it != j.end(); ++it) {
            const std::string& key = it.key();
            if (key == "__metadata__") {
                continue;
            }

            const auto& val = it.value();
            TensorInfo info;
            info.name = key;

            std::string dtype_str = val.value("dtype", "");
            if (dtype_str == "BF16") {
                info.dtype = DataType::BF16;
            } else if (dtype_str == "F16") {
                info.dtype = DataType::F16;
            } else if (dtype_str == "F32") {
                info.dtype = DataType::F32;
            } else {
                info.dtype = DataType::UNKNOWN;
            }

            if (val.contains("shape") && val["shape"].is_array()) {
                for (const auto& s : val["shape"]) {
                    info.shape.push_back(s.get<int64_t>());
                }
            }

            if (val.contains("data_offsets") && val["data_offsets"].is_array() && val["data_offsets"].size() == 2) {
                info.data_offsets[0] = val["data_offsets"][0].get<size_t>();
                info.data_offsets[1] = val["data_offsets"][1].get<size_t>();
            } else {
                std::cerr << "Missing or invalid data_offsets for tensor: " << key << std::endl;
                close();
                return false;
            }

            if (info.data_offsets[0] > info.data_offsets[1] || info.data_offsets[1] > total_tensor_buffer_len) {
                std::cerr << "Invalid data_offsets out of bounds for tensor: " << key << std::endl;
                close();
                return false;
            }

            size_t expected_nbytes = info.nbytes();
            size_t actual_nbytes = info.data_offsets[1] - info.data_offsets[0];
            if (expected_nbytes != actual_nbytes) {
                std::cerr << "Tensor size mismatch for " << key << ": expected " << expected_nbytes
                          << " but got " << actual_nbytes << std::endl;
                close();
                return false;
            }

            info.data = tensor_buffer_ + info.data_offsets[0];
            tensors_[key] = std::move(info);
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse safetensors JSON header: " << e.what() << std::endl;
        close();
        return false;
    }

    return true;
}

void SafetensorsLoader::close() {
    tensors_.clear();
    tensor_buffer_ = nullptr;
    header_size_ = 0;
    mapping_.reset();
}

const TensorInfo* SafetensorsLoader::get_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) {
        return &it->second;
    }
    return nullptr;
}

bool SafetensorsLoader::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

void SafetensorsLoader::bf16_to_fp32(const uint16_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t u32 = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &u32, sizeof(float));
    }
}

void SafetensorsLoader::f16_to_fp32(const uint16_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint16_t h = src[i];
        uint32_t sign = (static_cast<uint32_t>(h & 0x8000)) << 16;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f32_val = 0;

        if (exp == 0) {
            if (mant == 0) {
                f32_val = sign;
            } else {
                // Denormalized number
                while ((mant & 0x400) == 0) {
                    mant <<= 1;
                    exp--;
                }
                exp++;
                mant &= ~0x400;
                f32_val = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            // Inf or NaN
            f32_val = sign | 0x7F800000 | (mant << 13);
        } else {
            // Normalized number
            f32_val = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
        std::memcpy(&dst[i], &f32_val, sizeof(float));
    }
}

void SafetensorsLoader::bf16_to_f16(const uint16_t* src, uint16_t* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t u32 = static_cast<uint32_t>(src[i]) << 16;
        float f;
        std::memcpy(&f, &u32, sizeof(float));

        uint32_t f_bits;
        std::memcpy(&f_bits, &f, sizeof(float));

        uint32_t sign = (f_bits >> 16) & 0x8000;
        int32_t exp = static_cast<int32_t>((f_bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (f_bits >> 13) & 0x3FF;

        if (exp <= 0) {
            dst[i] = static_cast<uint16_t>(sign);
        } else if (exp >= 31) {
            dst[i] = static_cast<uint16_t>(sign | 0x7C00);
        } else {
            dst[i] = static_cast<uint16_t>(sign | (exp << 10) | mant);
        }
    }
}

} // namespace engine
