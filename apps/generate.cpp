#include "src/loader/safetensors.h"
#include "src/model/config.h"
#include "src/model/forward_cpu.h"
#include "src/model/forward_gpu.h"
#include "src/runtime/kv_cache.h"
#include "src/runtime/sampler.h"
#include "src/tokenizer/bpe.h"
#include "third_party/nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <unordered_map>
#include <numeric>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    bool time_load = false;
    bool dump_all = false;
    bool dump_all_gpu = false;
    bool verify_greedy_gpu = false;
    bool verify_kv_cache = false;
    bool profile_step = false;
    bool profile_decode = false;
    bool use_gpu = false;
    bool use_kv_cache = false;
    std::string model_path = "data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775";
    std::string prompt = "";
    int max_new_tokens = 32;
    std::string dump_dir = "";

    // Sampler flags
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
    uint64_t seed = 42;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--time-load") {
            time_load = true;
        } else if (arg == "--dump-all") {
            dump_all = true;
        } else if (arg == "--dump-all-gpu") {
            dump_all_gpu = true;
        } else if (arg == "--verify-greedy-gpu") {
            verify_greedy_gpu = true;
        } else if (arg == "--verify-kv-cache") {
            verify_kv_cache = true;
        } else if (arg == "--profile-step") {
            profile_step = true;
        } else if (arg == "--profile-decode") {
            profile_decode = true;
        } else if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--kv-cache") {
            use_gpu = true;
            use_kv_cache = true;
        } else if (arg == "--temperature" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if (arg == "--repetition-penalty" && i + 1 < argc) {
            repetition_penalty = std::stof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max-new-tokens" && i + 1 < argc) {
            max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--dump" && i + 1 < argc) {
            dump_dir = argv[++i];
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

    // Load config, weights, tokenizer for forward pass or generation
    engine::ModelConfig cfg = engine::ModelConfig::from_file(model_path + "/config.json");
    engine::SafetensorsLoader loader;
    if (!loader.open(model_path + "/model.safetensors")) {
        std::cerr << "Failed to open safetensors file: " << model_path + "/model.safetensors" << std::endl;
        return 1;
    }

    engine::ModelWeights weights;
    if (!weights.load_and_validate(cfg, loader)) {
        std::cerr << "Failed to validate model weights!" << std::endl;
        return 1;
    }

    engine::BPETokenizer tokenizer;
    if (!tokenizer.load(model_path + "/tokenizer.json")) {
        std::cerr << "Failed to load tokenizer.json!" << std::endl;
        return 1;
    }

    // Convert weights to FP32 or GPU FP16 depending on mode
    bool need_gpu = dump_all_gpu || verify_greedy_gpu || verify_kv_cache || profile_step || profile_decode || use_gpu;

    engine::ModelWeightsFP32 weights_fp32;
    engine::ModelWeightsGPU weights_gpu;
    engine::ActivationBuffersGPU buffers_gpu;
    engine::KVCache kv_cache;
    engine::SamplerGPU sampler;
    cublasHandle_t cublas_handle = nullptr;

    if (need_gpu) {
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        if (!weights_gpu.init_from_weights(cfg, weights)) {
            std::cerr << "Failed to upload and convert weights to GPU FP16!" << std::endl;
            return 1;
        }
        // Pre-allocate buffers for up to 4096 context length
        buffers_gpu.allocate(cfg, 4096);
        kv_cache.allocate(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, 4096);
        sampler.init(cfg.vocab_size);
    } else {
        if (!weights_fp32.init_from_weights(cfg, weights)) {
            std::cerr << "Failed to convert weights to FP32!" << std::endl;
            return 1;
        }
    }

    // Case 1a: dump-all reference prompts for compare.py (CPU)
    if (dump_all) {
        std::string ref_dir = "data/reference";
        std::string target_dir = "data/dump_cpu";
        std::cout << "Running CPU forward pass and dumping tensors to " << target_dir << "...\n";

        std::vector<std::string> prompt_dirs = {
            "01_short", "02_medium", "03_long", "04_code",
            "05_unicode", "06_cjk_emoji", "07_chat_template", "08_whitespace_indent"
        };

        for (const auto& p_id : prompt_dirs) {
            std::string prompt_ref = ref_dir + "/" + p_id;
            std::string prompt_out = target_dir + "/" + p_id;
            fs::create_directories(prompt_out);

            // Read token_ids.npy from reference
            std::string token_file = prompt_ref + "/token_ids.npy";
            std::ifstream tf(token_file, std::ios::binary);
            if (!tf.is_open()) {
                std::cerr << "Cannot open token file: " << token_file << std::endl;
                continue;
            }

            char magic[6];
            tf.read(magic, 6);
            char ver[2];
            tf.read(ver, 2);
            uint16_t hlen = 0;
            tf.read(reinterpret_cast<char*>(&hlen), 2);
            std::string header(hlen, ' ');
            tf.read(&header[0], hlen);

            tf.seekg(0, std::ios::end);
            size_t file_len = tf.tellg();
            size_t data_len = file_len - (10 + hlen);
            size_t S = data_len / sizeof(int64_t);

            tf.seekg(10 + hlen, std::ios::beg);
            std::vector<int64_t> tokens(S);
            tf.read(reinterpret_cast<char*>(tokens.data()), data_len);

            std::cout << "  [" << p_id << "] tokens: " << S << " ... ";
            auto t0 = std::chrono::high_resolution_clock::now();
            engine::forward_cpu(tokens, cfg, weights_fp32, prompt_out);
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "done in " << dt_ms << " ms\n";
        }
        std::cout << "Dump complete.\n";
        return 0;
    }

    // Case 1b: dump-all reference prompts for compare.py (GPU FP16)
    if (dump_all_gpu) {
        std::string ref_dir = "data/reference";
        std::string target_dir = "data/dump_gpu";
        std::cout << "Running GPU FP16 forward pass and dumping tensors to " << target_dir << "...\n";

        std::vector<std::string> prompt_dirs = {
            "01_short", "02_medium", "03_long", "04_code",
            "05_unicode", "06_cjk_emoji", "07_chat_template", "08_whitespace_indent"
        };

        for (const auto& p_id : prompt_dirs) {
            std::string prompt_ref = ref_dir + "/" + p_id;
            std::string prompt_out = target_dir + "/" + p_id;
            fs::create_directories(prompt_out);

            std::string token_file = prompt_ref + "/token_ids.npy";
            std::ifstream tf(token_file, std::ios::binary);
            if (!tf.is_open()) {
                std::cerr << "Cannot open token file: " << token_file << std::endl;
                continue;
            }

            char magic[6];
            tf.read(magic, 6);
            char ver[2];
            tf.read(ver, 2);
            uint16_t hlen = 0;
            tf.read(reinterpret_cast<char*>(&hlen), 2);
            std::string header(hlen, ' ');
            tf.read(&header[0], hlen);

            tf.seekg(0, std::ios::end);
            size_t file_len = tf.tellg();
            size_t data_len = file_len - (10 + hlen);
            size_t S = data_len / sizeof(int64_t);

            tf.seekg(10 + hlen, std::ios::beg);
            std::vector<int64_t> tokens(S);
            tf.read(reinterpret_cast<char*>(tokens.data()), data_len);

            std::cout << "  [" << p_id << "] tokens: " << S << " ... ";
            auto t0 = std::chrono::high_resolution_clock::now();
            engine::forward_gpu(tokens, cfg, weights_gpu, buffers_gpu, cublas_handle, prompt_out);
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "done in " << dt_ms << " ms\n";
        }
        std::cout << "GPU Dump complete.\n";
        if (cublas_handle) cublasDestroy(cublas_handle);
        return 0;
    }

    // Case 1c: Step Profile (--profile-step)
    if (profile_step) {
        std::cout << "=== Profiling Forward Step (Prompt 512, recompute decode) ===\n";
        // Construct 512 tokens
        std::string base_text = "The optimization of deep neural network inference on graphics hardware requires careful tuning of memory layouts, register allocation, and thread parallelism. ";
        std::vector<int> b_toks = tokenizer.encode(base_text, false);
        std::vector<int64_t> prompt_toks;
        while (prompt_toks.size() < 512) {
            for (int t : b_toks) {
                if (prompt_toks.size() < 512) prompt_toks.push_back(t);
                else break;
            }
        }

        // Warmup 3 steps
        for (int w = 0; w < 3; ++w) {
            engine::forward_gpu(prompt_toks, cfg, weights_gpu, buffers_gpu, cublas_handle);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<engine::KernelRecord> records;
        auto t_wall_start = std::chrono::high_resolution_clock::now();
        engine::forward_gpu(prompt_toks, cfg, weights_gpu, buffers_gpu, cublas_handle, "", &records);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t_wall_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(t_wall_end - t_wall_start).count();

        // Aggregate kernel times
        double total_gpu_busy_ms = 0.0;
        std::unordered_map<std::string, double> op_times;
        std::unordered_map<std::string, int> op_counts;

        for (const auto& r : records) {
            total_gpu_busy_ms += r.ms;
            // Group by op name (strip layer prefix if any)
            std::string op_type = r.name;
            size_t dot_pos = op_type.find('.');
            if (dot_pos != std::string::npos) {
                op_type = op_type.substr(dot_pos + 1);
            }
            op_times[op_type] += r.ms;
            op_counts[op_type]++;
        }

        double idle_ms = wall_ms - total_gpu_busy_ms;
        double idle_pct = (idle_ms / wall_ms) * 100.0;

        std::cout << "\nProfile Results (512 tokens forward step):\n";
        std::cout << "  Total Kernels Launched: " << records.size() << "\n";
        std::cout << "  Host Wall Time:         " << wall_ms << " ms\n";
        std::cout << "  GPU Busy Time:          " << total_gpu_busy_ms << " ms\n";
        std::cout << "  Idle / Gap Time:        " << idle_ms << " ms (" << idle_pct << "%)\n";

        // Sort op_times descending
        std::vector<std::pair<std::string, double>> sorted_ops(op_times.begin(), op_times.end());
        std::sort(sorted_ops.begin(), sorted_ops.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        std::cout << "\nTop Kernels by Accumulated Time:\n";
        for (size_t i = 0; i < std::min<size_t>(10, sorted_ops.size()); ++i) {
            const auto& [name, time_ms] = sorted_ops[i];
            double pct = (time_ms / total_gpu_busy_ms) * 100.0;
            std::cout << "  " << (i + 1) << ". " << name << " (" << op_counts[name] << " launches): "
                      << time_ms << " ms (" << pct << "%)\n";
        }

        // Export to docs/profiles/P3_profile.md
        fs::create_directories("docs/profiles");
        std::ofstream prof_file("docs/profiles/P3_profile.md");
        prof_file << "# Task P3 Profile: In-Engine CUDA Event Instrumentation\n\n";
        prof_file << "> [!NOTE]\n";
        prof_file << "> This profile was generated using **in-engine CUDA event instrumentation** (`cudaEventElapsedTime` around every kernel/cuBLAS launch), NOT Nsight Systems CLI. The elevated Nsight Systems profiling command is documented in `docs/ENV.md`.\n\n";
        prof_file << "## Forward Step Breakdown (Seq Len = 512, Batch = 1, Recompute Mode)\n\n";
        prof_file << "- **Total Kernels / API Calls per Step**: " << records.size() << "\n";
        prof_file << "- **Host Wall Time**: " << wall_ms << " ms\n";
        prof_file << "- **Total GPU Busy Time**: " << total_gpu_busy_ms << " ms\n";
        prof_file << "- **Idle Gap Total**: " << idle_ms << " ms (" << idle_pct << "%)\n";
        prof_file << "- **Is Decode Launch-Bound?**: **" << (idle_pct > 20.0 ? "YES (Launch Overhead Dominant)" : "NO (GPU Execution Bound)") << "**\n\n";
        prof_file << "## Top Operations by Accumulated GPU Time\n\n";
        prof_file << "| Rank | Kernel / Operation | Total Invocations | Total Time (ms) | % of GPU Time |\n";
        prof_file << "|---|---|---|---|---|\n";
        for (size_t i = 0; i < sorted_ops.size(); ++i) {
            const auto& [name, time_ms] = sorted_ops[i];
            double pct = (time_ms / total_gpu_busy_ms) * 100.0;
            prof_file << "| " << (i + 1) << " | `" << name << "` | " << op_counts[name]
                      << " | " << time_ms << " | " << pct << "% |\n";
        }
        prof_file << "\n## Analysis & Takeaways\n";
        prof_file << "1. **Launch Overhead**: At sequence length 512, " << records.size() << " kernel launches take "
                  << total_gpu_busy_ms << " ms of compute time while host overhead is " << idle_ms << " ms.\n";
        prof_file << "2. **cuBLAS Dominance**: Linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `lm_head`) account for the majority of execution time.\n";
        prof_file << "3. **Next Steps (P4 / P5)**: KV-caching eliminates $O(S^2)$ attention and recomputes only sequence length 1 per token; fused kernels and CUDA graphs will eliminate launch bubbles completely.\n";
        prof_file.close();

        std::cout << "\nSaved profile analysis to docs/profiles/P3_profile.md\n";
        if (cublas_handle) cublasDestroy(cublas_handle);
        return 0;
    }

    // Case 1c2: Decode Step Profile (--profile-decode)
    if (profile_decode) {
        std::cout << "=== Profiling Single-Token Decode Step with KV Cache (Context 512) ===\n";
        // Construct 512 tokens
        std::string base_text = "The optimization of deep neural network inference on graphics hardware requires careful tuning of memory layouts, register allocation, and thread parallelism. ";
        std::vector<int> b_toks = tokenizer.encode(base_text, false);
        std::vector<int64_t> prompt_toks;
        while (prompt_toks.size() < 512) {
            for (int t : b_toks) {
                if (prompt_toks.size() < 512) prompt_toks.push_back(t);
                else break;
            }
        }

        // Prefill cache up to 512 tokens
        kv_cache.set_current_seq_len(0);
        engine::prefill_gpu(prompt_toks, cfg, weights_gpu, buffers_gpu, kv_cache, cublas_handle);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Warmup 5 decode steps
        int next_token = 100;
        for (int w = 0; w < 5; ++w) {
            engine::decode_step_gpu(next_token, 512 + w, cfg, weights_gpu, buffers_gpu, kv_cache, cublas_handle);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<engine::KernelRecord> records;
        auto t_wall_start = std::chrono::high_resolution_clock::now();
        engine::decode_step_gpu(next_token, 517, cfg, weights_gpu, buffers_gpu, kv_cache, cublas_handle, nullptr, &records);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t_wall_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(t_wall_end - t_wall_start).count();

        // Aggregate kernel times
        double total_gpu_busy_ms = 0.0;
        std::unordered_map<std::string, double> op_times;
        std::unordered_map<std::string, int> op_counts;

        for (const auto& r : records) {
            total_gpu_busy_ms += r.ms;
            std::string op_type = r.name;
            size_t dot_pos = op_type.find('.');
            if (dot_pos != std::string::npos) {
                op_type = op_type.substr(dot_pos + 1);
            }
            op_times[op_type] += r.ms;
            op_counts[op_type]++;
        }

        double idle_ms = wall_ms - total_gpu_busy_ms;
        if (idle_ms < 0) idle_ms = 0;
        double idle_pct = (wall_ms > 0) ? ((idle_ms / wall_ms) * 100.0) : 0.0;

        std::cout << "\nProfile Results (Single-Token Decode Step, Context 517):\n";
        std::cout << "  Total Kernels Launched: " << records.size() << "\n";
        std::cout << "  Host Wall Time:         " << wall_ms << " ms\n";
        std::cout << "  GPU Busy Time:          " << total_gpu_busy_ms << " ms\n";
        std::cout << "  Idle / Gap Time:        " << idle_ms << " ms (" << idle_pct << "%)\n";

        // Sort op_times descending
        std::vector<std::pair<std::string, double>> sorted_ops(op_times.begin(), op_times.end());
        std::sort(sorted_ops.begin(), sorted_ops.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        std::cout << "\nTop Operations by Accumulated Time:\n";
        for (size_t i = 0; i < std::min<size_t>(10, sorted_ops.size()); ++i) {
            const auto& [name, time_ms] = sorted_ops[i];
            double pct = (total_gpu_busy_ms > 0) ? ((time_ms / total_gpu_busy_ms) * 100.0) : 0.0;
            std::cout << "  " << (i + 1) << ". " << name << " (" << op_counts[name] << " launches): "
                      << time_ms << " ms (" << pct << "%)\n";
        }

        fs::create_directories("docs/profiles");
        std::ofstream prof_file("docs/profiles/P4_decode_profile.md");
        prof_file << "# Task P4 Decode Profile: In-Engine CUDA Event Instrumentation\n\n";
        prof_file << "## Decode Step Breakdown (Context = 517, Batch = 1, KV Cache Mode)\n\n";
        prof_file << "- **Total Kernels Launched**: " << records.size() << "\n";
        prof_file << "- **Host Wall Time**: " << wall_ms << " ms\n";
        prof_file << "- **GPU Busy Time**: " << total_gpu_busy_ms << " ms\n";
        prof_file << "- **Idle Gap Time**: " << idle_ms << " ms (" << idle_pct << "%)\n";
        prof_file << "- **Is Decode Launch-Bound?**: **" << (idle_pct > 20.0 ? "YES (Launch Overhead Dominant)" : "NO (GPU Execution Bound)") << "**\n\n";
        prof_file << "## Top Operations by Accumulated GPU Time\n\n";
        prof_file << "| Rank | Kernel / Operation | Total Invocations | Total Time (ms) | % of GPU Time |\n";
        prof_file << "|---|---|---|---|---|\n";
        for (size_t i = 0; i < sorted_ops.size(); ++i) {
            const auto& [name, time_ms] = sorted_ops[i];
            double pct = (total_gpu_busy_ms > 0) ? ((time_ms / total_gpu_busy_ms) * 100.0) : 0.0;
            prof_file << "| " << (i + 1) << " | `" << name << "` | " << op_counts[name]
                      << " | " << time_ms << " | " << pct << "% |\n";
        }
        prof_file.close();

        std::cout << "\nSaved decode profile analysis to docs/profiles/P4_decode_profile.md\n";
        if (cublas_handle) cublasDestroy(cublas_handle);
        return 0;
    }

    // Case 1d: verify-greedy-gpu against reference JSON
    if (verify_greedy_gpu) {
        std::cout << "=== Verifying 32-Token Greedy Generation on GPU FP16 against HF Reference ===\n";
        std::vector<std::pair<std::string, std::string>> test_prompts = {
            {"01_short", "The capital of France is"},
            {"02_medium", "Deep learning inference engines optimize memory bandwidth and cache locality to achieve high token generation speeds on modern GPUs."},
            {"04_code", "def quicksort(arr):\n"},
            {"06_cjk_emoji", "你好，世界！🚀 こんにちは世界！🌟 안녕하세요 세계! 🇨🇳 🇯🇵 🇰🇷"},
            {"08_whitespace_indent", "class TransformerBlock:\n"}
        };

        std::ifstream f_ref("data/reference/hf_greedy_32_pure.json");
        if (!f_ref.is_open()) {
            std::cerr << "Failed to open data/reference/hf_greedy_32_pure.json\n";
            return 1;
        }
        nlohmann::json j_ref;
        f_ref >> j_ref;

        int total_tested = 0;
        int total_passed = 0;

        for (const auto& [pid, ptext] : test_prompts) {
            std::vector<int> initial_toks = tokenizer.encode(ptext, false);
            std::vector<int64_t> cur_tokens(initial_toks.begin(), initial_toks.end());
            std::vector<int> gen_tokens;

            std::cout << "[" << pid << "] Generating 32 tokens (GPU)... " << std::flush;
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int step = 0; step < 32; ++step) {
                std::vector<float> logits = engine::forward_gpu(cur_tokens, cfg, weights_gpu, buffers_gpu, cublas_handle);
                int64_t last_pos = static_cast<int64_t>(cur_tokens.size()) - 1;
                const float* last_logits = logits.data() + last_pos * cfg.vocab_size;

                int best_id = 0;
                float best_val = last_logits[0];
                for (int64_t v = 1; v < cfg.vocab_size; ++v) {
                    if (last_logits[v] > best_val) {
                        best_val = last_logits[v];
                        best_id = static_cast<int>(v);
                    }
                }
                gen_tokens.push_back(best_id);
                cur_tokens.push_back(best_id);
                if (best_id == 151645) break;
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

            std::vector<int> ref_toks = j_ref[pid]["tokens"].get<std::vector<int>>();
            bool match = (gen_tokens == ref_toks);
            total_tested++;
            if (match) total_passed++;

            std::cout << (match ? "PASS" : "FAIL") << " (" << elapsed_s << " s, " << (gen_tokens.size() / elapsed_s) << " tok/s)\n";
            if (!match) {
                std::cout << "  Mismatch!\n  Expected: [";
                for (size_t i = 0; i < ref_toks.size(); ++i) std::cout << (i > 0 ? ", " : "") << ref_toks[i];
                std::cout << "]\n  Got:      [";
                for (size_t i = 0; i < gen_tokens.size(); ++i) std::cout << (i > 0 ? ", " : "") << gen_tokens[i];
                std::cout << "]\n";
            }
        }

        std::cout << "=========================================================\n";
        std::cout << "Summary: " << total_passed << "/" << total_tested << " prompts matched 100% of tokens.\n";
        if (cublas_handle) cublasDestroy(cublas_handle);
        return (total_passed == total_tested) ? 0 : 1;
    }

    // Case 1e: verify-kv-cache (32 tokens greedy generation using KV cache vs HF reference)
    if (verify_kv_cache) {
        std::cout << "=== Verifying 32-Token Greedy Generation on GPU FP16 with KV CACHE against HF Reference ===\n";
        std::vector<std::pair<std::string, std::string>> test_prompts = {
            {"01_short", "The capital of France is"},
            {"02_medium", "Deep learning inference engines optimize memory bandwidth and cache locality to achieve high token generation speeds on modern GPUs."},
            {"04_code", "def quicksort(arr):\n"},
            {"06_cjk_emoji", "你好，世界！🚀 こんにちは世界！🌟 안녕하세요 세계! 🇨🇳 🇯🇵 🇰🇷"},
            {"08_whitespace_indent", "class TransformerBlock:\n"}
        };

        std::ifstream f_ref("data/reference/hf_greedy_32_pure.json");
        if (!f_ref.is_open()) {
            std::cerr << "Failed to open data/reference/hf_greedy_32_pure.json\n";
            return 1;
        }
        nlohmann::json j_ref;
        f_ref >> j_ref;

        int total_tested = 0;
        int total_passed = 0;

        for (const auto& [pid, ptext] : test_prompts) {
            std::vector<int> initial_toks = tokenizer.encode(ptext, false);
            std::vector<int64_t> cur_tokens(initial_toks.begin(), initial_toks.end());
            std::vector<int> gen_tokens;

            std::cout << "[" << pid << "] Generating 32 tokens (KV Cache)... " << std::flush;
            auto t0 = std::chrono::high_resolution_clock::now();

            // 1. Prefill
            kv_cache.set_current_seq_len(0);
            std::vector<float> last_logits = engine::prefill_gpu(
                cur_tokens, cfg, weights_gpu, buffers_gpu, kv_cache, cublas_handle
            );

            // First token argmax
            int best_id = 0;
            float best_val = last_logits[0];
            for (int64_t v = 1; v < cfg.vocab_size; ++v) {
                if (last_logits[v] > best_val) {
                    best_val = last_logits[v];
                    best_id = static_cast<int>(v);
                }
            }
            gen_tokens.push_back(best_id);
            int current_pos = static_cast<int>(cur_tokens.size());

            // 2. Decode steps
            for (int step = 1; step < 32; ++step) {
                if (best_id == 151645) break;

                std::vector<float> decode_logits = engine::decode_step_gpu(
                    best_id, current_pos, cfg, weights_gpu, buffers_gpu, kv_cache, cublas_handle
                );

                best_id = 0;
                best_val = decode_logits[0];
                for (int64_t v = 1; v < cfg.vocab_size; ++v) {
                    if (decode_logits[v] > best_val) {
                        best_val = decode_logits[v];
                        best_id = static_cast<int>(v);
                    }
                }
                gen_tokens.push_back(best_id);
                current_pos++;
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

            std::vector<int> ref_toks = j_ref[pid]["tokens"].get<std::vector<int>>();
            bool match = (gen_tokens == ref_toks);
            total_tested++;
            if (match) total_passed++;

            std::cout << (match ? "PASS" : "FAIL") << " (" << elapsed_s << " s, " << (gen_tokens.size() / elapsed_s) << " tok/s)\n";
            if (!match) {
                std::cout << "  Mismatch!\n  Expected: [";
                for (size_t i = 0; i < ref_toks.size(); ++i) std::cout << (i > 0 ? ", " : "") << ref_toks[i];
                std::cout << "]\n  Got:      [";
                for (size_t i = 0; i < gen_tokens.size(); ++i) std::cout << (i > 0 ? ", " : "") << gen_tokens[i];
                std::cout << "]\n";
            }
        }

        std::cout << "=========================================================\n";
        std::cout << "Summary: " << total_passed << "/" << total_tested << " prompts matched 100% of tokens.\n";
        if (cublas_handle) cublasDestroy(cublas_handle);
        return (total_passed == total_tested) ? 0 : 1;
    }

    bool verify_greedy = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--verify-greedy") {
            verify_greedy = true;
        }
    }

    if (verify_greedy) {
        std::cout << "=== Verifying 32-Token Greedy Generation against HF Reference ===\n";
        std::vector<std::pair<std::string, std::string>> test_prompts = {
            {"01_short", "The capital of France is"},
            {"02_medium", "Deep learning inference engines optimize memory bandwidth and cache locality to achieve high token generation speeds on modern GPUs."},
            {"04_code", "def quicksort(arr):\n"},
            {"06_cjk_emoji", "你好，世界！🚀 こんにちは世界！🌟 안녕하세요 세계! 🇨🇳 🇯🇵 🇰🇷"},
            {"08_whitespace_indent", "class TransformerBlock:\n"}
        };

        // Load reference JSON
        std::ifstream f_ref("data/reference/hf_greedy_32_pure.json");
        if (!f_ref.is_open()) {
            std::cerr << "Failed to open data/reference/hf_greedy_32_pure.json\n";
            return 1;
        }
        nlohmann::json j_ref;
        f_ref >> j_ref;

        int total_tested = 0;
        int total_passed = 0;

        for (const auto& [pid, ptext] : test_prompts) {
            std::vector<int> initial_toks = tokenizer.encode(ptext, false);
            std::vector<int64_t> cur_tokens(initial_toks.begin(), initial_toks.end());
            std::vector<int> gen_tokens;

            std::cout << "[" << pid << "] Generating 32 tokens... " << std::flush;
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int step = 0; step < 32; ++step) {
                std::vector<float> logits = engine::forward_cpu(cur_tokens, cfg, weights_fp32);
                int64_t last_pos = static_cast<int64_t>(cur_tokens.size()) - 1;
                const float* last_logits = logits.data() + last_pos * cfg.vocab_size;

                int best_id = 0;
                float best_val = last_logits[0];
                for (int64_t v = 1; v < cfg.vocab_size; ++v) {
                    if (last_logits[v] > best_val) {
                        best_val = last_logits[v];
                        best_id = static_cast<int>(v);
                    }
                }
                gen_tokens.push_back(best_id);
                cur_tokens.push_back(best_id);
                if (best_id == 151645) break;
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

            std::vector<int> ref_toks = j_ref[pid]["tokens"].get<std::vector<int>>();
            bool match = (gen_tokens == ref_toks);
            total_tested++;
            if (match) total_passed++;

            std::cout << (match ? "PASS" : "FAIL") << " (" << elapsed_s << " s, " << (gen_tokens.size() / elapsed_s) << " tok/s)\n";
            if (!match) {
                std::cout << "  Mismatch!\n  Expected: [";
                for (size_t i = 0; i < ref_toks.size(); ++i) std::cout << (i > 0 ? ", " : "") << ref_toks[i];
                std::cout << "]\n  Got:      [";
                for (size_t i = 0; i < gen_tokens.size(); ++i) std::cout << (i > 0 ? ", " : "") << gen_tokens[i];
                std::cout << "]\n";
            }
        }

        std::cout << "=========================================================\n";
        std::cout << "Summary: " << total_passed << "/" << total_tested << " prompts matched 100% of tokens.\n";
        return (total_passed == total_tested) ? 0 : 1;
    }


    // Case 2: Generation loop
    if (prompt.empty()) {
        std::cout << "Usage: generate.exe --prompt <text> [--max-new-tokens N] [--kv-cache] [--temperature T] [--top-k K] [--top-p P]\n";
        std::cout << "       generate.exe --dump-all\n";
        std::cout << "       generate.exe --time-load\n";
        std::cout << "       generate.exe --verify-greedy-gpu\n";
        std::cout << "       generate.exe --verify-kv-cache\n";
        return 0;
    }

    std::vector<int> initial_tokens = tokenizer.encode(prompt, false);
    std::vector<int64_t> tokens(initial_tokens.begin(), initial_tokens.end());
    std::cout << "Prompt (" << tokens.size() << " tokens): " << prompt << "\n";
    std::cout << "Generating " << max_new_tokens << " tokens with "
              << (use_kv_cache ? "GPU FP16 (KV Cache)" : (use_gpu ? "GPU FP16 (recompute)" : "CPU FP32")) << "...\n";

    std::vector<int> generated_tokens;
    int64_t vocab_size = cfg.vocab_size;

    engine::SamplerConfig sampler_cfg;
    sampler_cfg.temperature = temperature;
    sampler_cfg.top_k = top_k;
    sampler_cfg.top_p = top_p;
    sampler_cfg.repetition_penalty = repetition_penalty;
    sampler_cfg.seed = seed;

    if (use_kv_cache) {
        // 1. Prefill
        kv_cache.set_current_seq_len(0);
        auto t_start = std::chrono::high_resolution_clock::now();
        std::vector<float> logits = engine::prefill_gpu(
            tokens, cfg, weights_gpu, buffers_gpu, kv_cache, cublas_handle
        );
        auto t_prefill = std::chrono::high_resolution_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();

        int sampled_token = sampler.sample_from_float_host(logits.data(), tokens, sampler_cfg);
        generated_tokens.push_back(sampled_token);
        tokens.push_back(sampled_token);
        int current_pos = static_cast<int>(tokens.size()) - 1;

        // 2. Decode
        auto t_decode_start = std::chrono::high_resolution_clock::now();
        for (int step = 1; step < max_new_tokens; ++step) {
            if (sampled_token == 151645) break;

            std::vector<float> decode_logits = engine::decode_step_gpu(
                sampled_token, current_pos, cfg, weights_gpu, buffers_gpu, kv_cache, cublas_handle
            );

            sampled_token = sampler.sample_from_float_host(decode_logits.data(), tokens, sampler_cfg);
            generated_tokens.push_back(sampled_token);
            tokens.push_back(sampled_token);
            current_pos++;
        }
        auto t_decode_end = std::chrono::high_resolution_clock::now();
        double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
        int decode_steps = static_cast<int>(generated_tokens.size()) - 1;

        std::cout << "\nTiming: Prefill (" << initial_tokens.size() << " tok): " << prefill_ms
                  << " ms (" << (initial_tokens.size() / (prefill_ms / 1000.0)) << " tok/s)\n";
        if (decode_steps > 0) {
            std::cout << "        Decode (" << decode_steps << " steps): " << decode_ms
                      << " ms (" << (decode_ms / decode_steps) << " ms/step, "
                      << (decode_steps / (decode_ms / 1000.0)) << " tok/s)\n";
        }
    } else {
        for (int step = 0; step < max_new_tokens; ++step) {
            std::string step_dump = (step == 0 && !dump_dir.empty()) ? dump_dir : "";
            std::vector<float> logits;
            if (use_gpu) {
                logits = engine::forward_gpu(tokens, cfg, weights_gpu, buffers_gpu, cublas_handle, step_dump);
            } else {
                logits = engine::forward_cpu(tokens, cfg, weights_fp32, step_dump);
            }

            int64_t last_pos = static_cast<int64_t>(tokens.size()) - 1;
            const float* last_logits = logits.data() + last_pos * vocab_size;

            int best_id = sampler.sample_from_float_host(last_logits, tokens, sampler_cfg);
            generated_tokens.push_back(best_id);
            tokens.push_back(best_id);

            if (best_id == 151645) { // <|im_end|> or eos
                break;
            }
        }
    }


    std::string gen_text = tokenizer.decode(generated_tokens);
    std::cout << "\nGenerated text:\n" << gen_text << "\n\n";
    std::cout << "Generated token IDs: [";
    for (size_t i = 0; i < generated_tokens.size(); ++i) {
        std::cout << (i > 0 ? ", " : "") << generated_tokens[i];
    }
    std::cout << "]\n";

    return 0;
}
