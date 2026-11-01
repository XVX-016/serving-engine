// tokenize_stdin: reads one JSON-encoded string per line from stdin,
// writes one JSON array of token ids per line to stdout.
// Protocol: each input line is: "<json_escaped_string>\n"
// Each output line is: [id1,id2,...]\n
// A blank input line exits the process.
// This lets Python call us in batch mode without per-string process overhead.

#include "src/tokenizer/bpe.h"
#include "third_party/nlohmann/json.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const std::string tokenizer_path = argc > 1 ? argv[1]
        : "data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json";

    engine::BPETokenizer tok;
    if (!tok.load(tokenizer_path)) {
        std::cerr << "ERROR: failed to load tokenizer from " << tokenizer_path << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) break;  // blank line = done

        // Each line is a JSON string (quoted, escaped)
        std::string text;
        try {
            nlohmann::json j = nlohmann::json::parse(line);
            if (!j.is_string()) {
                std::cout << "null\n";
                continue;
            }
            text = j.get<std::string>();
        } catch (...) {
            std::cout << "null\n";
            continue;
        }

        std::vector<int> ids = tok.encode(text, /*allowed_special=*/true);

        // Output as compact JSON array
        nlohmann::json out = ids;
        std::cout << out.dump() << "\n";
        std::cout.flush();
    }
    return 0;
}

