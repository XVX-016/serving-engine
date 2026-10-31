#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace engine {

struct ChatMessage {
    std::string role;
    std::string content;
};

class BPETokenizer {
public:
    BPETokenizer() = default;
    ~BPETokenizer() = default;

    // Load from HF tokenizer.json
    bool load(const std::string& tokenizer_json_path);

    // Encode text to token IDs
    std::vector<int> encode(const std::string& text, bool allowed_special = true) const;

    // Decode token IDs back to UTF-8 text
    std::string decode(const std::vector<int>& ids, bool skip_special_tokens = false) const;

    // Apply Qwen chat template
    std::string apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt = true) const;

    // Special token IDs
    int im_start_id() const { return im_start_id_; }
    int im_end_id() const { return im_end_id_; }
    int endoftext_id() const { return endoftext_id_; }

    size_t vocab_size() const { return vocab_to_id_.size(); }

private:
    void init_byte_mappings();
    std::vector<std::string> split_pre_tokenizer(const std::string& text) const;
    std::vector<int> bpe(const std::string& token_str) const;

    std::unordered_map<std::string, int> vocab_to_id_;
    std::vector<std::string> id_to_vocab_;
    std::unordered_map<std::string, int> bpe_ranks_; // pair "part1 part2" -> rank

    // Added tokens / special tokens
    std::unordered_map<std::string, int> special_token_to_id_;
    std::unordered_map<int, std::string> id_to_special_token_;

    // GPT-2 byte encoder / decoder
    std::unordered_map<uint8_t, std::string> byte_to_unicode_;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    int im_start_id_ = 151644;
    int im_end_id_ = 151645;
    int endoftext_id_ = 151643;
};

} // namespace engine
