#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "third_party/doctest.h"
#include "src/tokenizer/bpe.h"
#include "third_party/nlohmann/json.hpp"

#include <iostream>
#include <fstream>

TEST_CASE("BPE Tokenizer Corpus Verification") {
    const std::string tokenizer_json_path = "data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json";
    const std::string corpus_json_path = "data/tokenizer_corpus.json";

    engine::BPETokenizer tokenizer;
    bool loaded = tokenizer.load(tokenizer_json_path);
    REQUIRE(loaded);
    CHECK(tokenizer.vocab_size() >= 151643);

    // Read corpus
    std::ifstream f(corpus_json_path);
    REQUIRE(f.is_open());
    nlohmann::json corpus;
    f >> corpus;
    REQUIRE(corpus.is_array());
    REQUIRE(corpus.size() >= 300);

    std::cout << "\n=== Running BPETokenizer on " << corpus.size() << " corpus test cases ===" << std::endl;

    size_t passed_tokens = 0;
    size_t passed_decodes = 0;

    for (const auto& item : corpus) {
        int id = item["id"].get<int>();
        std::string text = item["text"].get<std::string>();
        std::vector<int> expected_ids = item["token_ids"].get<std::vector<int>>();
        std::string expected_decoded = item["decoded"].get<std::string>();

        // 1. Test Encode
        std::vector<int> actual_ids = tokenizer.encode(text, true);
        if (actual_ids != expected_ids) {
            std::cerr << "Mismatch token_ids for item #" << id << ":\n  Text: " << text << "\n  Expected: [";
            for (size_t i = 0; i < expected_ids.size(); ++i) std::cerr << (i > 0 ? ", " : "") << expected_ids[i];
            std::cerr << "]\n  Actual:   [";
            for (size_t i = 0; i < actual_ids.size(); ++i) std::cerr << (i > 0 ? ", " : "") << actual_ids[i];
            std::cerr << "]\n";
        }
        CHECK(actual_ids == expected_ids);
        if (actual_ids == expected_ids) passed_tokens++;

        // 2. Test Decode
        std::string actual_decoded = tokenizer.decode(actual_ids, false);
        if (actual_decoded != expected_decoded) {
            std::cerr << "Mismatch decoded for item #" << id << ":\n  Expected: " << expected_decoded
                      << "\n  Actual:   " << actual_decoded << "\n";
        }
        CHECK(actual_decoded == expected_decoded);
        if (actual_decoded == expected_decoded) passed_decodes++;
    }

    std::cout << "Summary: " << passed_tokens << "/" << corpus.size() << " token ID matches, "
              << passed_decodes << "/" << corpus.size() << " decode round-trip matches.\n";

    // 3. Test chat template
    std::vector<engine::ChatMessage> msgs = {
        {"system", "You are a helpful assistant."},
        {"user", "Hello!"}
    };
    std::string templated = tokenizer.apply_chat_template(msgs, true);
    std::string expected_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n";
    CHECK(templated == expected_template);
}
