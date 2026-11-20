with open('apps/generate.cpp', 'r', encoding='utf-8') as f:
    text = f.read()

# Add #include <cuda_profiler_api.h> if not present
if '#include <cuda_profiler_api.h>' not in text:
    text = '#include <cuda_profiler_api.h>\n' + text

target_launch = '''        std::cout << "Launching 1 targeted graph step for profiling...\\n";
        decode_graph.launch(next_token, 134, buffers_gpu);
        CUDA_CHECK(cudaDeviceSynchronize());'''

repl_launch = '''        std::cout << "Launching 1 targeted graph step for profiling...\\n";
        cudaProfilerStart();
        decode_graph.launch(next_token, 134, buffers_gpu);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaProfilerStop();'''

text = text.replace(target_launch, repl_launch, 1)

with open('apps/generate.cpp', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated generate.cpp with cudaProfilerStart/Stop')
