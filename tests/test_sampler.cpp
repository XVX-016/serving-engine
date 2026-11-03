#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "third_party/doctest.h"
#include "src/runtime/sampler.h"
#include "src/common/cuda_utils.h"

#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <map>
#include <unordered_set>

using namespace engine;

TEST_CASE("Sampler GPU: Argmax (Temperature = 0)") {
    const int vocab_size = 151936;
    SamplerGPU sampler(vocab_size);

    std::vector<float> logits(vocab_size, -10.0f);
    int target_token = 42123;
    logits[target_token] = 55.5f;

    SamplerConfig cfg;
    cfg.temperature = 0.0f;

    int sampled = sampler.sample_from_float_host(logits.data(), {}, cfg);
    CHECK(sampled == target_token);

    // Another token
    target_token = 151643;
    logits[42123] = -10.0f;
    logits[target_token] = 100.0f;
    sampled = sampler.sample_from_float_host(logits.data(), {}, cfg);
    CHECK(sampled == target_token);
}

TEST_CASE("Sampler GPU: Top-K and Top-P filtering never emit excluded tokens") {
    const int vocab_size = 151936;
    SamplerGPU sampler(vocab_size);

    std::vector<float> logits(vocab_size, -100.0f);
    // Allow top 5 tokens with high logits
    std::vector<int> allowed_tokens = {10, 20, 30, 40, 50};
    for (int t : allowed_tokens) {
        logits[t] = 10.0f;
    }

    SamplerConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k = 5;
    cfg.top_p = 1.0f;

    std::unordered_set<int> allowed_set(allowed_tokens.begin(), allowed_tokens.end());

    // Sample 1000 times with random seeds
    for (uint64_t seed = 1; seed <= 1000; ++seed) {
        cfg.seed = seed;
        int sampled = sampler.sample_from_float_host(logits.data(), {}, cfg);
        CHECK(allowed_set.count(sampled) == 1);
    }
}

TEST_CASE("Sampler GPU: Repetition penalty correctly suppresses repeats") {
    const int vocab_size = 151936;
    SamplerGPU sampler(vocab_size);

    std::vector<float> logits(vocab_size, 0.0f);
    int token_a = 100;
    int token_b = 200;
    logits[token_a] = 10.0f;
    logits[token_b] = 8.0f;

    SamplerConfig cfg;
    cfg.temperature = 0.0f; // Argmax
    cfg.repetition_penalty = 1.0f;

    // Without penalty: token_a wins
    int sampled = sampler.sample_from_float_host(logits.data(), {}, cfg);
    CHECK(sampled == token_a);

    // With penalty on token_a: logit(token_a) = 10.0 / 2.0 = 5.0 < 8.0
    cfg.repetition_penalty = 2.0f;
    sampled = sampler.sample_from_float_host(logits.data(), {token_a}, cfg);
    CHECK(sampled == token_b);
}

TEST_CASE("Sampler GPU: Statistical test (100k draws within 3-sigma)") {
    const int vocab_size = 151936;
    SamplerGPU sampler(vocab_size);

    // Multinomial distribution over 4 tokens with known target probabilities:
    // P = [0.4, 0.3, 0.2, 0.1]
    // logit = log(P) + C
    std::vector<float> target_p = {0.40f, 0.30f, 0.20f, 0.10f};
    std::vector<int> tokens = {1000, 2000, 3000, 4000};

    std::vector<float> logits(vocab_size, -100.0f);
    for (size_t i = 0; i < tokens.size(); ++i) {
        logits[tokens[i]] = std::log(target_p[i]);
    }

    SamplerConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k = 4;
    cfg.top_p = 1.0f;
    cfg.seed = 424242;

    const int num_draws = 100000;
    std::vector<int> counts(tokens.size(), 0);

    for (int i = 0; i < num_draws; ++i) {
        // RNG advances internally, pass seed = 0 so seed is not re-set
        cfg.seed = (i == 0) ? 424242 : 0;
        int sampled = sampler.sample_from_float_host(logits.data(), {}, cfg);
        for (size_t t = 0; t < tokens.size(); ++t) {
            if (sampled == tokens[t]) {
                counts[t]++;
                break;
            }
        }
    }

    std::cout << "\n=== Sampler 100k Statistical Verification Results ===" << std::endl;
    for (size_t t = 0; t < tokens.size(); ++t) {
        double expected = target_p[t] * num_draws;
        double empirical = counts[t];
        double p = target_p[t];
        double sigma = std::sqrt(num_draws * p * (1.0 - p));
        double delta = std::abs(empirical - expected);
        double num_sigma = delta / sigma;

        std::cout << "Token " << tokens[t] << ": Expected=" << expected
                  << ", Observed=" << empirical
                  << ", Delta=" << delta
                  << ", Sigma=" << sigma
                  << " (" << num_sigma << " sigma)" << std::endl;

        // Gate: within 3-sigma confidence bounds
        CHECK(num_sigma < 3.0);
    }
}

