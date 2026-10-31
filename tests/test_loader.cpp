#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "third_party/doctest.h"
#include "src/loader/safetensors.h"
#include "src/model/config.h"

#include <iostream>
#include <cmath>

TEST_CASE("Safetensors Loader & Config Parser") {
    const std::string safetensors_path = "data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors";
    const std::string config_path = "data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/config.json";

    // 1. Config Parsing
    engine::ModelConfig cfg = engine::ModelConfig::from_file(config_path);
    CHECK(cfg.vocab_size == 151936);
    CHECK(cfg.hidden_size == 896);
    CHECK(cfg.intermediate_size == 4864);
    CHECK(cfg.num_hidden_layers == 24);
    CHECK(cfg.num_attention_heads == 14);
    CHECK(cfg.num_key_value_heads == 2);
    CHECK(cfg.head_dim == 64);
    CHECK(cfg.tie_word_embeddings == true);

    // 2. Open Safetensors
    engine::SafetensorsLoader loader;
    bool opened = loader.open(safetensors_path);
    REQUIRE(opened);
    CHECK(loader.header_size() > 0);
    CHECK(loader.file_size() > 900 * 1024 * 1024);

    // 3. Inspect and Print Tensors
    const auto& tensors = loader.tensors();
    CHECK(tensors.size() == 290); // 1 embed + 24 * 12 + 1 norm

    std::cout << "\n=== Safetensors Loader Checked " << tensors.size() << " tensors ===" << std::endl;
    std::cout << "Header size: " << loader.header_size() << " bytes, Total file: " << loader.file_size() << " bytes\n";

    // Check specific essential tensors
    CHECK(loader.has_tensor("model.embed_tokens.weight"));
    const auto* embed = loader.get_tensor("model.embed_tokens.weight");
    REQUIRE(embed != nullptr);
    CHECK(embed->dtype == engine::DataType::BF16);
    CHECK(embed->shape == std::vector<int64_t>{151936, 896});
    CHECK(embed->numel() == 151936 * 896);
    CHECK(embed->nbytes() == 151936 * 896 * 2);

    CHECK(loader.has_tensor("model.norm.weight"));
    const auto* norm = loader.get_tensor("model.norm.weight");
    REQUIRE(norm != nullptr);
    CHECK(norm->shape == std::vector<int64_t>{896});

    // 4. Validate ModelWeights binding & shape checking
    engine::ModelWeights weights;
    bool validated = weights.load_and_validate(cfg, loader);
    CHECK(validated);
    CHECK(weights.embed_tokens != nullptr);
    CHECK(weights.final_norm != nullptr);
    CHECK(weights.lm_head == weights.embed_tokens); // Tied embeddings check
    CHECK(weights.layers.size() == 24);

    for (int i = 0; i < 24; ++i) {
        CHECK(weights.layers[i].input_layernorm != nullptr);
        CHECK(weights.layers[i].post_attention_layernorm != nullptr);
        CHECK(weights.layers[i].q_proj_weight != nullptr);
        CHECK(weights.layers[i].q_proj_bias != nullptr);
        CHECK(weights.layers[i].k_proj_weight != nullptr);
        CHECK(weights.layers[i].k_proj_bias != nullptr);
        CHECK(weights.layers[i].v_proj_weight != nullptr);
        CHECK(weights.layers[i].v_proj_bias != nullptr);
        CHECK(weights.layers[i].o_proj_weight != nullptr);
        CHECK(weights.layers[i].gate_proj_weight != nullptr);
        CHECK(weights.layers[i].up_proj_weight != nullptr);
        CHECK(weights.layers[i].down_proj_weight != nullptr);
    }

    // 5. Test conversions
    uint16_t bf16_test = 0x3F80; // 1.0f in BF16
    float fp32_res = 0.0f;
    engine::SafetensorsLoader::bf16_to_fp32(&bf16_test, &fp32_res, 1);
    CHECK(fp32_res == 1.0f);

    uint16_t f16_test = 0x3C00; // 1.0f in FP16
    engine::SafetensorsLoader::f16_to_fp32(&f16_test, &fp32_res, 1);
    CHECK(fp32_res == 1.0f);

    uint16_t f16_out = 0;
    engine::SafetensorsLoader::bf16_to_f16(&bf16_test, &f16_out, 1);
    CHECK(f16_out == 0x3C00);
}
