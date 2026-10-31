#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>
#include <memory>

namespace engine {

enum class DataType {
    BF16,
    F16,
    F32,
    UNKNOWN
};

struct TensorInfo {
    std::string name;
    DataType dtype = DataType::UNKNOWN;
    std::vector<int64_t> shape;
    size_t data_offsets[2] = {0, 0}; // start, end within binary buffer
    const void* data = nullptr;      // mapped pointer directly into memory
    size_t nbytes() const;
    int64_t numel() const;
};

class IFileMapping {
public:
    virtual ~IFileMapping() = default;
    virtual const uint8_t* data() const = 0;
    virtual size_t size() const = 0;
};

class SafetensorsLoader {
public:
    SafetensorsLoader() = default;
    ~SafetensorsLoader() = default;

    bool open(const std::string& filepath);
    void close();

    const std::unordered_map<std::string, TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo* get_tensor(const std::string& name) const;
    bool has_tensor(const std::string& name) const;

    size_t header_size() const { return header_size_; }
    size_t file_size() const { return mapping_ ? mapping_->size() : 0; }

    // Conversion utilities
    static void bf16_to_fp32(const uint16_t* src, float* dst, size_t count);
    static void f16_to_fp32(const uint16_t* src, float* dst, size_t count);
    static void bf16_to_f16(const uint16_t* src, uint16_t* dst, size_t count);

private:
    std::unique_ptr<IFileMapping> mapping_;
    size_t header_size_ = 0;
    const uint8_t* tensor_buffer_ = nullptr;
    std::unordered_map<std::string, TensorInfo> tensors_;
};

} // namespace engine
