with open('apps/generate.cpp', 'r', encoding='utf-8') as f:
    text = f.read()

target = '        std::cout << "Graph step steady-state 2-event time: " << unprofiled_gpu_ms << " ms\\n";'
repl = '''        std::cout << "Captured CUDA Graph Node Count: " << decode_graph.node_count() << "\\n";
        std::cout << "Graph step steady-state 2-event time: " << unprofiled_gpu_ms << " ms\\n";'''

text = text.replace(target, repl, 1)

with open('apps/generate.cpp', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated apps/generate.cpp with node count print')
