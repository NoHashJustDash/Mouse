// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "engine/nn/cached_backend.h"
#include "engine/nn/legal_evaluator.h"
class Provider final : public mouse::nn::Backend {
   public:
    std::size_t calls = 0;
    mouse::nn::ModelMetadata model;
    mouse::nn::BackendInfo identity;
    Provider() {
        model.input_format = 1;
        identity.cache_namespace = std::string(64, 'a');
    }
    [[nodiscard]] const mouse::nn::ModelMetadata& metadata() const noexcept override {
        return model;
    }
    [[nodiscard]] const mouse::nn::BackendInfo& info() const noexcept override { return identity; }
    mouse::Result<void> evaluate(std::span<const mouse::nn::InputPlanes>,
                                 std::span<mouse::nn::NNOutput> out) override {
        ++calls;
        for (auto& value : out) {
            for (std::size_t i = 0; i < value.policy_logits.size(); ++i)
                value.policy_logits.at(i) = static_cast<float>(i) * 0.123456791F;
            value.wdl = {0.2F, 0.3F, 0.5F};
            value.moves_left = 10.125F;
        }
        return {};
    }
};
int main() {
    try {
        using namespace mouse::nn;
        for (const std::size_t batch : {std::size_t{1}, std::size_t{8}, std::size_t{64}}) {
            auto physical = std::make_unique<Provider>();
            auto* before_provider = physical.get();
            auto raw = create_cached_backend(std::move(physical), 1U << 20U, 64);
            auto old_capacity = NNCache::create(1U << 20U, std::string(64, 'a'));
            if (!raw || !old_capacity) return 1;
            const auto capacity = (*old_capacity)->stats().capacity;
            auto cache = LegalCache::create(capacity, std::string(64, 'a'));
            if (!cache) return 2;
            Provider after_provider;
            LegalEvaluator legal(after_provider, 64, cache->get());
            std::vector<InputPlanes> inputs(batch);
            std::vector<PolicyIndices> policies(batch);
            std::vector<NNOutput> raw_output(batch);
            std::vector<LegalOutput> legal_output(batch);
            for (std::size_t row = 0; row < batch; ++row) {
                inputs.at(row)[0].mask = row + 1;
                inputs.at(row)[0].value = 1.0F;
                inputs.at(row)[109].value = static_cast<float>(row);
                policies.at(row).size = 20;
                for (std::size_t j = 0; j < 20; ++j)
                    policies.at(row).values.at(j) = static_cast<std::uint16_t>(j * 97);
            }
            auto before = [&] { return (*raw)->evaluate(inputs, raw_output); };
            auto after = [&] { return legal.evaluate(inputs, policies, legal_output); };
            if (!before() || !after()) return 3;
            std::array<double, 5> old{}, now{};
            auto measure = [&](auto&& evaluate) {
                const auto start = std::chrono::steady_clock::now();
                for (unsigned iteration = 0; iteration < 10000; ++iteration)
                    if (!evaluate()) throw std::runtime_error("evaluation failure");
                return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() -
                                                                start)
                           .count() /
                       (10000.0 * static_cast<double>(batch));
            };
            for (std::size_t round = 0; round < 5; ++round) {
                if (round % 2 == 0) {
                    old.at(round) = measure(before);
                    now.at(round) = measure(after);
                } else {
                    now.at(round) = measure(after);
                    old.at(round) = measure(before);
                }
            }
            for (std::size_t row = 0; row < batch; ++row) {
                if (raw_output.at(row).wdl != legal_output.at(row).wdl ||
                    raw_output.at(row).moves_left != legal_output.at(row).moves_left)
                    return 4;
                for (std::size_t j = 0; j < 20; ++j)
                    if (raw_output.at(row).policy_logits.at(policies.at(row).values.at(j)) !=
                        legal_output.at(row).logits.at(j))
                        return 5;
            }
            if (before_provider->calls != 1 || after_provider.calls != 1) return 6;
            std::ranges::sort(old);
            std::ranges::sort(now);
            std::cout << "CACHE_CPU_HITS batch=" << batch << " capacity=" << capacity
                      << " median_raw_ns_per_row=" << old[2]
                      << " median_legal_ns_per_row=" << now[2] << " ratio=" << old[2] / now[2]
                      << " warm_provider_calls=1/1 exact_legal_heads=true\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Cache benchmark failed: " << error.what() << '\n';
        return 7;
    } catch (...) {
        std::cerr << "Cache benchmark failed with a non-standard exception\n";
        return 8;
    }
}
