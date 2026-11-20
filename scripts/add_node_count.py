with open('src/model/forward_gpu.h', 'r', encoding='utf-8') as f:
    text = f.read()

target = '    bool is_captured() const { return captured_; }'
repl = '''    bool is_captured() const { return captured_; }
    size_t node_count() const {
        if (!graph_) return 0;
        size_t num_nodes = 0;
        cudaGraphGetNodes(graph_, nullptr, &num_nodes);
        return num_nodes;
    }'''

text = text.replace(target, repl, 1)

with open('src/model/forward_gpu.h', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated forward_gpu.h with node_count()')
