#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace wl {

struct Vec3 {
    double x{};
    double y{};
    double z{};
};

struct Box {
    Vec3 lengths{};
    bool periodic{true};
};

struct Geometry {
    std::vector<Vec3> positions;
    std::vector<Vec3> axes;
    Box box;

    [[nodiscard]] std::size_t size() const noexcept { return positions.size(); }
    void validate() const;
    static Geometry simple_cubic(std::size_t nx, std::size_t ny, std::size_t nz,
                                 double spacing = 1.0, Vec3 axis = {0.0, 0.0, 1.0},
                                 bool periodic = true);
    static Geometry load_xyz_axes(const std::string& path, Box box);
    static Geometry load_csv(const std::string& path, Box box);
};

[[nodiscard]] Vec3 minimum_image(Vec3 displacement, const Box& box);
[[nodiscard]] double dipolar_pair(Vec3 displacement, Vec3 axis_i, Vec3 axis_j,
                                  double coupling_scale);

class Couplings {
public:
    virtual ~Couplings() = default;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    [[nodiscard]] virtual double at(std::size_t i, std::size_t j) const noexcept = 0;
    virtual void add_flip_delta(std::size_t flipped, std::int8_t old_spin,
                                std::span<double> fields) const = 0;
    [[nodiscard]] virtual std::string backend_name() const = 0;
};

class DenseCouplings final : public Couplings {
public:
    DenseCouplings(const Geometry& geometry, double coupling_scale);
    [[nodiscard]] std::size_t size() const noexcept override { return n_; }
    [[nodiscard]] double at(std::size_t i, std::size_t j) const noexcept override;
    void add_flip_delta(std::size_t flipped, std::int8_t old_spin,
                        std::span<double> fields) const override;
    [[nodiscard]] std::string backend_name() const override { return "dense"; }
private:
    std::size_t n_{};
    std::vector<double> matrix_;
};

class CsrCouplings final : public Couplings {
public:
    CsrCouplings(const Geometry& geometry, double coupling_scale, double cutoff);
    [[nodiscard]] std::size_t size() const noexcept override { return n_; }
    [[nodiscard]] double at(std::size_t i, std::size_t j) const noexcept override;
    void add_flip_delta(std::size_t flipped, std::int8_t old_spin,
                        std::span<double> fields) const override;
    [[nodiscard]] std::string backend_name() const override { return "csr"; }
private:
    std::size_t n_{};
    std::vector<std::size_t> offsets_;
    std::vector<std::size_t> neighbors_;
    std::vector<double> values_;
};

[[nodiscard]] std::vector<double> local_fields(const Couplings& couplings,
                                                std::span<const std::int8_t> spins);
[[nodiscard]] double total_energy(const Couplings& couplings,
                                  std::span<const std::int8_t> spins);
[[nodiscard]] double flip_delta(std::size_t i, std::span<const std::int8_t> spins,
                                std::span<const double> fields);

} // namespace wl
