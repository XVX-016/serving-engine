#include "src/loader/safetensors.h"
#include "src/model/config.h"
#include "src/tokenizer/bpe.h"

#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <cstring>

int main(int argc, char** argv) {
    bool time_load = false;
    std::string model_path = "data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--time-load") {
            time_load = true;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
    }

    if (time_load) {
        std::cout << "=========================================================\n";
        std::cout << " Cold-Start Load Time Breakdown (Process Start to Host)  \n";
        std::cout << "=========================================================\n";

        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Config Loading
        auto t_cfg_start = std::chrono::high_resolution_clock::now();
        engine::ModelConfig cfg = engine::ModelConfig::from_file(model_path + "/config.json");
        auto t_cfg_end = std::chrono::high_resolution_clock::now();
        double dt_cfg_ms = std::chrono::duration<double, std::milli>(t_cfg_end - t_cfg_start).count();

        // 2. Open & Memory Map Safetensors
        auto t_map_start = std::chrono::high_resolution_clock::now();
        engine::SafetensorsLoader loader;
        if (!loader.open(model_path + "/model.safetensors")) {
            std::cerr << "Failed to open safetensors file!" << std::endl;
            return 1;
        }
        auto t_map_end = std::chrono::high_resolution_clock::now();
        double dt_map_ms = std::chrono::duration<double, std::milli>(t_map_end - t_map_start).count();

        // 3. Parse and Validate Model Weights Struct
        auto t_weights_start = std::chrono::high_resolution_clock::now();
        engine::ModelWeights weights;
        if (!weights.load_and_validate(cfg, loader)) {
            std::cerr << "Failed to validate model weights!" << std::endl;
            return 1;
        }
        auto t_weights_end = std::chrono::high_resolution_clock::now();
        double dt_weights_ms = std::chrono::duration<double, std::milli>(t_weights_end - t_weights_start).count();

        // 4. Tokenizer Loading
        auto t_tok_start = std::chrono::high_resolution_clock::now();
        engine::BPETokenizer tokenizer;
        if (!tokenizer.load(model_path + "/tokenizer.json")) {
            std::cerr << "Failed to load tokenizer.json!" << std::endl;
            return 1;
        }
        auto t_tok_end = std::chrono::high_resolution_clock::now();
        double dt_tok_ms = std::chrono::duration<double, std::milli>(t_tok_end - t_tok_start).count();

        // 5. Weight Conversion Test (Host BF16 to Host FP16 / FP32)
        auto t_conv_start = std::chrono::high_resolution_clock::now();
        // Allocate host buffer for all weights to measure full conversion throughput
        size_t total_elements = 0;
        for (const auto& [name, info] : loader.tensors()) {
            total_elements += static_cast<size_t>(info.numel());
        }
        std::vector<uint16_t> fp16_weights(total_elements);
        size_t offset = 0;
        for (const auto& [name, info] : loader.tensors()) {
            if (info.dtype == engine::DataType::BF16) {
                engine::SafetensorsLoader::bf16_to_f16(
                    static_cast<const uint16_t*>(info.data),
                    fp16_weights.data() + offset,
                    static_cast<size_t>(info.numel())
                );
            }
            offset += static_cast<size_t>(info.numel());
        }
        auto t_conv_end = std::chrono::high_resolution_clock::now();
        double dt_conv_ms = std::chrono::duration<double, std::milli>(t_conv_end - t_conv_start).count();

        auto t_total_end = std::chrono::high_resolution_clock::now();
        double dt_total_ms = std::chrono::duration<double, std::milli>(t_total_end - t0).count();

        std::cout << "Model:            Qwen2.5-0.5B-Instruct (" << (loader.file_size() / (1024.0 * 1024.0)) << " MB safetensors)\n";
        std::cout << "Tensors:          " << loader.tensors().size() << " tensors, " << total_elements << " parameters\n";
        std::cout << "---------------------------------------------------------\n";
        std::cout << "1. Config Parse:      " << dt_cfg_ms << " ms\n";
        std::cout << "2. Open & Mmap file:  " << dt_map_ms << " ms (header: " << (loader.header_size() / 1024.0) << " KB)\n";
        std::cout << "3. Bind & Validate:   " << dt_weights_ms << " ms\n";
        std::cout << "4. Tokenizer Load:    " << dt_tok_ms << " ms (vocab: " << tokenizer.vocab_size() << ")\n";
        std::cout << "5. BF16->FP16 Convert:" << dt_conv_ms << " ms (" << (total_elements * 2.0 / (1024.0 * 1024.0)) / (dt_conv_ms / 1000.0) << " MB/s)\n";
        std::cout << "---------------------------------------------------------\n";
        std::cout << "Total Load Time:      " << dt_total_ms << " ms\n";
        std::cout << "=========================================================\n";
        return 0;
    }

    std::cout << "Serving Engine CLI. Pass --time-load to measure load latency.\n";
    return 0;
}
