#include "src/tokenizer/bpe.h"
#include "src/tokenizer/unicode_tables.h"
#include "third_party/nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cctype>

namespace engine {

// UTF-8 decode helper: returns codepoint and length in bytes
static inline uint32_t decode_utf8(const char* ptr, size_t max_len, size_t& len) {
    if (max_len == 0) { len = 0; return 0; }
    uint8_t c = static_cast<uint8_t>(ptr[0]);
    if (c < 0x80) {
        len = 1;
        return c;
    } else if ((c & 0xE0) == 0xC0) {
        if (max_len < 2) { len = 1; return c; }
        len = 2;
        return ((c & 0x1F) << 6) | (static_cast<uint8_t>(ptr[1]) & 0x3F);
    } else if ((c & 0xF0) == 0xE0) {
        if (max_len < 3) { len = 1; return c; }
        len = 3;
        return ((c & 0x0F) << 12) | ((static_cast<uint8_t>(ptr[1]) & 0x3F) << 6) | (static_cast<uint8_t>(ptr[2]) & 0x3F);
    } else if ((c & 0xF8) == 0xF0) {
        if (max_len < 4) { len = 1; return c; }
        len = 4;
        return ((c & 0x07) << 18) | ((static_cast<uint8_t>(ptr[1]) & 0x3F) << 12) |
               ((static_cast<uint8_t>(ptr[2]) & 0x3F) << 6) | (static_cast<uint8_t>(ptr[3]) & 0x3F);
    }
    len = 1;
    return c;
}

static inline std::string encode_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return s;
}

void BPETokenizer::init_byte_mappings() {
    byte_to_unicode_.clear();
    unicode_to_byte_.clear();

    // GPT-2 byte encoder mapping
    // Bytes that map to themselves:
    // '!' (33) to '~' (126), '¡' (161) to '¬' (172), '®' (174) to 'ÿ' (255)
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        uint8_t b = static_cast<uint8_t>(bs[i]);
        uint32_t cp = static_cast<uint32_t>(cs[i]);
        std::string u = encode_utf8(cp);
        byte_to_unicode_[b] = u;
        unicode_to_byte_[u] = b;
    }
}

bool BPETokenizer::load(const std::string& tokenizer_json_path) {
    init_byte_mappings();

    std::ifstream f(tokenizer_json_path);
    if (!f.is_open()) {
        std::cerr << "Failed to open tokenizer.json: " << tokenizer_json_path << std::endl;
        return false;
    }

    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse tokenizer.json: " << e.what() << std::endl;
        return false;
    }

    // 1. Load Vocab
    vocab_to_id_.clear();
    if (!j.contains("model") || !j["model"].contains("vocab")) {
        std::cerr << "tokenizer.json missing model.vocab" << std::endl;
        return false;
    }

    const auto& vocab = j["model"]["vocab"];
    int max_id = -1;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        int id = it.value().get<int>();
        vocab_to_id_[it.key()] = id;
        if (id > max_id) max_id = id;
    }

    id_to_vocab_.resize(max_id + 1);
    for (const auto& [token, id] : vocab_to_id_) {
        id_to_vocab_[id] = token;
    }

    // 2. Load Merges
    bpe_ranks_.clear();
    if (j["model"].contains("merges")) {
        int rank = 0;
        for (const auto& m : j["model"]["merges"]) {
            std::string merge_str = m.get<std::string>();
            bpe_ranks_[merge_str] = rank++;
        }
    }

    // 3. Load added_tokens / special tokens
    special_token_to_id_.clear();
    id_to_special_token_.clear();
    if (j.contains("added_tokens")) {
        for (const auto& at : j["added_tokens"]) {
            std::string content = at["content"].get<std::string>();
            int id = at["id"].get<int>();
            special_token_to_id_[content] = id;
            id_to_special_token_[id] = content;
            vocab_to_id_[content] = id;
            if (id >= static_cast<int>(id_to_vocab_.size())) {
                id_to_vocab_.resize(id + 1);
            }
            id_to_vocab_[id] = content;
        }
    }

    // Well-known special token IDs
    if (special_token_to_id_.count("<|im_start|>")) im_start_id_ = special_token_to_id_["<|im_start|>"];
    if (special_token_to_id_.count("<|im_end|>")) im_end_id_ = special_token_to_id_["<|im_end|>"];
    if (special_token_to_id_.count("<|endoftext|>")) endoftext_id_ = special_token_to_id_["<|endoftext|>"];

    return true;
}

// Helper: codepoint array representation of UTF-8 string
struct CharInfo {
    uint32_t cp;
    size_t byte_len;
    bool is_l;
    bool is_n;
    bool is_s;
};

// Hand-written matcher implementing exact regex:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
static size_t match_regex_token(const std::vector<CharInfo>& chars, size_t i) {
    size_t n = chars.size();
    if (i >= n) return 0;

    const auto& c0 = chars[i];

    // Branch 0: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c0.cp == '\'') {
        size_t rem = n - i - 1;
        if (rem >= 2) {
            uint32_t c1 = chars[i + 1].cp;
            uint32_t c2 = chars[i + 2].cp;
            char s2[3] = { static_cast<char>(std::tolower(c1)), static_cast<char>(std::tolower(c2)), 0 };
            if (std::string(s2) == "re" || std::string(s2) == "ve" || std::string(s2) == "ll") {
                return 3;
            }
        }
        if (rem >= 1) {
            char c1 = static_cast<char>(std::tolower(chars[i + 1].cp));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                return 2;
            }
        }
    }

    // Branch 1: [^\r\n\p{L}\p{N}]?\p{L}+
    if (c0.cp != '\r' && c0.cp != '\n' && !c0.is_l && !c0.is_n) {
        if (i + 1 < n && chars[i + 1].is_l) {
            size_t j = i + 1;
            while (j < n && chars[j].is_l) {
                j++;
            }
            return j - i;
        }
    } else if (c0.is_l) {
        size_t j = i + 1;
        while (j < n && chars[j].is_l) {
            j++;
        }
        return j - i;
    }

    // Branch 2: \p{N} (single number)
    if (c0.is_n) {
        return 1;
    }

    // Branch 3:  ?[^\s\p{L}\p{N}]+[\r\n]*
    bool has_leading_space = (c0.cp == ' ');
    size_t start_sym = has_leading_space ? (i + 1) : i;
    if (start_sym < n && !chars[start_sym].is_s && !chars[start_sym].is_l && !chars[start_sym].is_n) {
        size_t j = start_sym + 1;
        while (j < n && !chars[j].is_s && !chars[j].is_l && !chars[j].is_n) {
            j++;
        }
        while (j < n && (chars[j].cp == '\r' || chars[j].cp == '\n')) {
            j++;
        }
        return j - i;
    }

    // Whitespace branches
    if (c0.is_s) {
        // Scan forward through all consecutive whitespace
        size_t j = i;
        size_t last_nl = 0;
        bool has_nl = false;
        while (j < n && chars[j].is_s) {
            if (chars[j].cp == '\r' || chars[j].cp == '\n') {
                size_t k = j;
                while (k < n && (chars[k].cp == '\r' || chars[k].cp == '\n')) {
                    k++;
                }
                last_nl = k;
                has_nl = true;
                j = k;
            } else {
                j++;
            }
        }

        // Branch 4: \s*[\r\n]+
        if (has_nl) {
            return last_nl - i;
        }

        size_t ws_len = j - i;

        // Branch 5: \s+(?!\S)
        if (j == n) {
            return ws_len;
        } else if (ws_len >= 2) {
            return ws_len - 1;
        }

        // Branch 6: \s+
        return ws_len;
    }

    // Fallback: 1 char
    return 1;
}

std::vector<std::string> BPETokenizer::split_pre_tokenizer(const std::string& text) const {
    std::vector<CharInfo> chars;
    chars.reserve(text.size());

    size_t pos = 0;
    while (pos < text.size()) {
        size_t b_len = 0;
        uint32_t cp = decode_utf8(text.data() + pos, text.size() - pos, b_len);
        CharInfo ci;
        ci.cp = cp;
        ci.byte_len = b_len;
        ci.is_l = unicode_is_letter(cp);
        ci.is_n = unicode_is_number(cp);
        ci.is_s = unicode_is_whitespace(cp);
        chars.push_back(ci);
        pos += b_len;
    }

    std::vector<std::string> pieces;
    size_t i = 0;
    while (i < chars.size()) {
        size_t num_chars = match_regex_token(chars, i);
        if (num_chars == 0) num_chars = 1;

        std::string piece;
        for (size_t k = 0; k < num_chars; ++k) {
            piece.append(encode_utf8(chars[i + k].cp));
        }
        pieces.push_back(std::move(piece));
        i += num_chars;
    }

    return pieces;
}

std::vector<int> BPETokenizer::bpe(const std::string& token_str) const {
    // Map each byte to its Unicode representation in the GPT-2 byte alphabet
    std::vector<std::string> word;
    word.reserve(token_str.size());
    for (uint8_t b : token_str) {
        auto it = byte_to_unicode_.find(b);
        if (it != byte_to_unicode_.end()) {
            word.push_back(it->second);
        } else {
            word.push_back(std::string(1, static_cast<char>(b)));
        }
    }

    if (word.size() <= 1) {
        std::vector<int> res;
        for (const auto& w : word) {
            auto it = vocab_to_id_.find(w);
            if (it != vocab_to_id_.end()) res.push_back(it->second);
        }
        return res;
    }

    // Merge iteratively using rank table
    while (word.size() > 1) {
        int min_rank = 1000000000;
        size_t min_idx = 0;
        bool found = false;

        for (size_t i = 0; i < word.size() - 1; ++i) {
            std::string pair = word[i] + " " + word[i + 1];
            auto it = bpe_ranks_.find(pair);
            if (it != bpe_ranks_.end()) {
                if (it->second < min_rank) {
                    min_rank = it->second;
                    min_idx = i;
                    found = true;
                }
            }
        }

        if (!found) {
            break;
        }

        // Merge all occurrences of this winning pair
        std::string first = word[min_idx];
        std::string second = word[min_idx + 1];
        std::vector<std::string> new_word;
        new_word.reserve(word.size());

        size_t i = 0;
        while (i < word.size()) {
            if (i < word.size() - 1 && word[i] == first && word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }
        word = std::move(new_word);
    }

    std::vector<int> token_ids;
    token_ids.reserve(word.size());
    for (const auto& w : word) {
        auto it = vocab_to_id_.find(w);
        if (it != vocab_to_id_.end()) {
            token_ids.push_back(it->second);
        }
    }

    return token_ids;
}

std::vector<int> BPETokenizer::encode(const std::string& text, bool allowed_special) const {
    std::vector<int> result;

    if (text.empty()) {
        return result;
    }

    // If allowed_special is true, find any occurrences of special tokens first
    if (allowed_special && !special_token_to_id_.empty()) {
        size_t pos = 0;
        while (pos < text.size()) {
            // Find earliest matching special token
            size_t earliest_pos = std::string::npos;
            std::string matched_token;
            int matched_id = -1;

            for (const auto& [token, id] : special_token_to_id_) {
                size_t p = text.find(token, pos);
                if (p != std::string::npos) {
                    if (earliest_pos == std::string::npos || p < earliest_pos ||
                        (p == earliest_pos && token.size() > matched_token.size())) {
                        earliest_pos = p;
                        matched_token = token;
                        matched_id = id;
                    }
                }
            }

            if (earliest_pos != std::string::npos) {
                // Encode everything before special token
                if (earliest_pos > pos) {
                    std::string segment = text.substr(pos, earliest_pos - pos);
                    std::vector<std::string> pieces = split_pre_tokenizer(segment);
                    for (const auto& p : pieces) {
                        auto ids = bpe(p);
                        result.insert(result.end(), ids.begin(), ids.end());
                    }
                }
                // Add special token ID
                result.push_back(matched_id);
                pos = earliest_pos + matched_token.size();
            } else {
                // No more special tokens
                std::string segment = text.substr(pos);
                std::vector<std::string> pieces = split_pre_tokenizer(segment);
                for (const auto& p : pieces) {
                    auto ids = bpe(p);
                    result.insert(result.end(), ids.begin(), ids.end());
                }
                break;
            }
        }
    } else {
        std::vector<std::string> pieces = split_pre_tokenizer(text);
        for (const auto& p : pieces) {
            auto ids = bpe(p);
            result.insert(result.end(), ids.begin(), ids.end());
        }
    }

    return result;
}

std::string BPETokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    std::string byte_stream;

    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(id_to_vocab_.size())) {
            continue;
        }

        if (id_to_special_token_.count(id)) {
            if (!skip_special_tokens) {
                byte_stream += id_to_special_token_.at(id);
            }
            continue;
        }

        const std::string& token_str = id_to_vocab_[id];
        // Convert Unicode symbols back to original bytes
        size_t pos = 0;
        while (pos < token_str.size()) {
            size_t b_len = 0;
            decode_utf8(token_str.data() + pos, token_str.size() - pos, b_len);
            std::string u = token_str.substr(pos, b_len);
            pos += b_len;

            auto it = unicode_to_byte_.find(u);
            if (it != unicode_to_byte_.end()) {
                byte_stream.push_back(static_cast<char>(it->second));
            } else {
                byte_stream += u;
            }
        }
    }

    return byte_stream;
}

std::string BPETokenizer::apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt) const {
    std::ostringstream oss;

    // Check if first message is system
    bool has_system = !messages.empty() && messages[0].role == "system";
    if (has_system) {
        oss << "<|im_start|>system\n" << messages[0].content << "<|im_end|>\n";
    } else {
        oss << "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n";
    }

    size_t start_idx = has_system ? 1 : 0;
    for (size_t i = start_idx; i < messages.size(); ++i) {
        oss << "<|im_start|>" << messages[i].role << "\n"
            << messages[i].content << "<|im_end|>\n";
    }

    if (add_generation_prompt) {
        oss << "<|im_start|>assistant\n";
    }

    return oss.str();
}

} // namespace engine
