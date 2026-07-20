#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <span>

namespace wl {

class Xoshiro256StarStar {
public:
    explicit Xoshiro256StarStar(std::uint64_t seed = 1, std::uint64_t stream = 0) {
        seed_state(seed, stream);
    }

    [[nodiscard]] std::uint64_t operator()() noexcept {
        const auto result = rotl(state_[1] * 5, 7) * 9;
        const auto t = state_[1] << 17;
        state_[2] ^= state_[0]; state_[3] ^= state_[1];
        state_[1] ^= state_[2]; state_[0] ^= state_[3];
        state_[2] ^= t; state_[3] = rotl(state_[3], 45);
        return result;
    }

    [[nodiscard]] double uniform() noexcept {
        return static_cast<double>((*this)() >> 11) * 0x1.0p-53;
    }

    [[nodiscard]] std::uint64_t bounded(std::uint64_t bound) noexcept {
        if (bound == 0) return 0;
        const auto threshold = static_cast<std::uint64_t>(-bound) % bound;
        for (;;) {
            const auto x = (*this)();
            if (x >= threshold) return x % bound;
        }
    }

    [[nodiscard]] const std::array<std::uint64_t, 4>& state() const noexcept { return state_; }
    void set_state(std::array<std::uint64_t, 4> value) noexcept { state_ = value; }

private:
    std::array<std::uint64_t, 4> state_{};
    static constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }
    static std::uint64_t splitmix(std::uint64_t& x) noexcept {
        auto z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    void seed_state(std::uint64_t seed, std::uint64_t stream) noexcept {
        std::uint64_t x = seed ^ (0xd2b74407b1ce6e93ULL * (stream + 1));
        for (auto& word : state_) word = splitmix(x);
    }
};

} // namespace wl

