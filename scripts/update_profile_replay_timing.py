with open('apps/generate.cpp', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        std::cout << "Launching 1 targeted graph step for profiling...\\n";
        cudaProfilerStart();
        decode_graph.launch(next_token, 134, buffers_gpu);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaProfilerStop();
        std::cout << "Done profile_graph_step.\\n";'''

repl = '''        std::cout << "Launching targeted graph step for profiling with precise events...\\n";
        cudaEvent_t ev_prof_start, ev_prof_stop;
        CUDA_CHECK(cudaEventCreate(&ev_prof_start));
        CUDA_CHECK(cudaEventCreate(&ev_prof_stop));

        cudaProfilerStart();
        CUDA_CHECK(cudaEventRecord(ev_prof_start, decode_graph.stream()));
        decode_graph.launch(next_token, 134, buffers_gpu);
        CUDA_CHECK(cudaEventRecord(ev_prof_stop, decode_graph.stream()));
        CUDA_CHECK(cudaEventSynchronize(ev_prof_stop));
        cudaProfilerStop();

        float profiled_replay_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&profiled_replay_ms, ev_prof_start, ev_prof_stop));
        std::cout << "Graph step exact profiled replay time: " << profiled_replay_ms << " ms\\n";
        CUDA_CHECK(cudaEventDestroy(ev_prof_start));
        CUDA_CHECK(cudaEventDestroy(ev_prof_stop));
        std::cout << "Done profile_graph_step.\\n";'''

text = text.replace(target, repl, 1)

with open('apps/generate.cpp', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated apps/generate.cpp with exact profiled replay event timing')
