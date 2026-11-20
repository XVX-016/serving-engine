with open('apps/generate.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

cli_target = '} else if (arg == "--profile-decode") {\n            profile_decode = true;'
cli_repl = '} else if (arg == "--profile-decode") {\n            profile_decode = true;\n        } else if (arg == "--profile-graph-step") {\n            profile_graph_step = true;'

content = content.replace(cli_target, cli_repl, 1)

gpu_target = 'profile_decode || bench_enqueue'
gpu_repl = 'profile_decode || profile_graph_step || bench_enqueue'
content = content.replace(gpu_target, gpu_repl, 1)

handler_block = '''    // Profile Single Fused CUDA Graph Step at Context 128 for NCU / Profiling (--profile-graph-step)
    if (profile_graph_step) {
        std::cout << "=== Preparing Fused CUDA Graph Step at Context 128 ===\\n";
        std::string base_text = "The optimization of deep neural network inference on graphics hardware requires careful tuning of memory layouts, register allocation, and thread parallelism. ";
        std::vector<int> b_toks = tokenizer.encode(base_text, false);
        std::vector<int64_t> prompt_toks;
        while (prompt_toks.size() < 128) {
            for (int t : b_toks) {
                if (prompt_toks.size() < 128) prompt_toks.push_back(t);
                else break;
            }
        }

        // Prefill cache up to 128 tokens
        kv_cache.set_current_seq_len(0);
        engine::prefill_gpu(prompt_toks, cfg, weights_gpu, buffers_gpu, kv_cache, cublas_handle);
        CUDA_CHECK(cudaDeviceSynchronize());

        engine::DecodeCudaGraph decode_graph;
        decode_graph.capture(cfg, weights_gpu, buffers_gpu, kv_cache, cublas_handle,
                             buffers_gpu.logits.data(), use_legacy_gemv);
        CUDA_CHECK(cudaDeviceSynchronize());

        int next_token = 100;
        // Warmup 5 graph launches
        for (int w = 0; w < 5; ++w) {
            decode_graph.launch(next_token, 128 + w, buffers_gpu);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Measure unprofiled steady-state step time with 2 events around the whole step
        cudaEvent_t ev_step_start, ev_step_stop;
        CUDA_CHECK(cudaEventCreate(&ev_step_start));
        CUDA_CHECK(cudaEventCreate(&ev_step_stop));
        CUDA_CHECK(cudaEventRecord(ev_step_start, decode_graph.stream()));
        decode_graph.launch(next_token, 133, buffers_gpu);
        CUDA_CHECK(cudaEventRecord(ev_step_stop, decode_graph.stream()));
        CUDA_CHECK(cudaEventSynchronize(ev_step_stop));
        float unprofiled_gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&unprofiled_gpu_ms, ev_step_start, ev_step_stop));
        CUDA_CHECK(cudaEventDestroy(ev_step_start));
        CUDA_CHECK(cudaEventDestroy(ev_step_stop));

        std::cout << "Graph step steady-state 2-event time: " << unprofiled_gpu_ms << " ms\\n";
        std::cout << "Launching 1 targeted graph step for profiling...\\n";
        decode_graph.launch(next_token, 134, buffers_gpu);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "Done profile_graph_step.\\n";

        if (cublas_handle) cublasDestroy(cublas_handle);
        return 0;
    }

'''

target_handler_loc = '    // Case 1c2: Decode Step Profile (--profile-decode)'
content = content.replace(target_handler_loc, handler_block + target_handler_loc, 1)

with open('apps/generate.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated apps/generate.cpp successfully.')
