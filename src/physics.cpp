#include "wl/physics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <cctype>

namespace wl {
namespace {

double dot(Vec3 a, Vec3 b) noexcept { return a.x*b.x + a.y*b.y + a.z*b.z; }
double norm(Vec3 a) noexcept { return std::sqrt(dot(a, a)); }
Vec3 normalized(Vec3 a) {
    const auto length = norm(a);
    if (!(length > 0.0) || !std::isfinite(length))
        throw std::invalid_argument("Ising axes must be finite non-zero vectors");
    return {a.x/length, a.y/length, a.z/length};
}
Vec3 subtract(Vec3 a, Vec3 b) noexcept { return {a.x-b.x, a.y-b.y, a.z-b.z}; }

double pair_for(const Geometry& geometry, std::size_t i, std::size_t j, double scale) {
    auto displacement = subtract(geometry.positions[j], geometry.positions[i]);
    displacement = minimum_image(displacement, geometry.box);
    return dipolar_pair(displacement, geometry.axes[i], geometry.axes[j], scale);
}

} // namespace

void Geometry::validate() const {
    if (positions.empty()) throw std::invalid_argument("Geometry contains no sites");
    if (positions.size() != axes.size())
        throw std::invalid_argument("Geometry positions/axes size mismatch");
    if (!order_weights.empty() && order_weights.size() != positions.size())
        throw std::invalid_argument("Geometry order-parameter weight count mismatch");
    if (box.periodic) {
        const auto valid_length = [](double length) {
            return std::isfinite(length) && length >= 0.0;
        };
        if (!valid_length(box.lengths.x) || !valid_length(box.lengths.y) ||
            !valid_length(box.lengths.z))
            throw std::invalid_argument("Periodic box lengths must be finite and non-negative");
        if (box.lengths.x == 0.0 && box.lengths.y == 0.0 && box.lengths.z == 0.0)
            throw std::invalid_argument("At least one periodic box length must be positive");
    }
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto& p = positions[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            throw std::invalid_argument("Site coordinates must be finite");
        const auto length = norm(axes[i]);
        if (std::abs(length - 1.0) > 1e-10)
            throw std::invalid_argument("Ising axes must be normalized");
        if (!order_weights.empty() && !std::isfinite(order_weights[i]))
            throw std::invalid_argument("Order-parameter weights must be finite");
    }
}

Geometry Geometry::simple_cubic(std::size_t nx, std::size_t ny, std::size_t nz,
                                double spacing, Vec3 axis, bool periodic) {
    if (nx == 0 || ny == 0 || nz == 0 || !(spacing > 0.0))
        throw std::invalid_argument("Invalid simple-cubic dimensions or spacing");
    Geometry result;
    result.box = {{static_cast<double>(nx)*spacing,static_cast<double>(ny)*spacing,
                   static_cast<double>(nz)*spacing}, periodic};
    axis = normalized(axis);
    result.positions.reserve(nx*ny*nz);
    result.axes.reserve(nx*ny*nz);
    for (std::size_t z = 0; z < nz; ++z)
        for (std::size_t y = 0; y < ny; ++y)
            for (std::size_t x = 0; x < nx; ++x) {
                result.positions.push_back({static_cast<double>(x)*spacing,
                                            static_cast<double>(y)*spacing,
                                            static_cast<double>(z)*spacing});
                result.axes.push_back(axis);
            }
    result.validate();
    return result;
}

Geometry Geometry::load_xyz_axes(const std::string& path, Box box) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open geometry file: " + path);
    Geometry result;
    result.box = box;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream values(line);
        Vec3 p, a;
        if (!(values >> p.x >> p.y >> p.z >> a.x >> a.y >> a.z))
            throw std::runtime_error("Invalid geometry line " + std::to_string(line_number));
        result.positions.push_back(p);
        result.axes.push_back(normalized(a));
    }
    result.validate();
    return result;
}

Geometry Geometry::load_csv(const std::string& path, Box box) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open CSV geometry file: " + path);
    auto trim = [](std::string value) {
        const auto first = value.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return std::string{};
        const auto last = value.find_last_not_of(" \t\r\n");
        value = value.substr(first, last-first+1);
        if (value.size() >= 3 && static_cast<unsigned char>(value[0]) == 0xef &&
            static_cast<unsigned char>(value[1]) == 0xbb && static_cast<unsigned char>(value[2]) == 0xbf)
            value.erase(0,3);
        return value;
    };
    auto columns = [&](const std::string& line) {
        std::vector<std::string> result; std::stringstream stream(line); std::string item;
        while (std::getline(stream,item,',')) result.push_back(trim(item));
        return result;
    };
    Geometry result; result.box=box;
    std::string line; std::size_t line_number=0; bool header_read=false;
    bool have_order_weights=false;
    while (std::getline(input,line)) {
        ++line_number; const auto clean=trim(line);
        if (clean.empty() || clean[0]=='#') continue;
        const auto fields=columns(clean);
        if (!header_read) {
            const std::array<std::string,6> expected{"x","y","z","mx","my","mz"};
            if (fields.size()!=expected.size() && fields.size()!=expected.size()+1)
                throw std::runtime_error("CSV geometry header must be x,y,z,mx,my,mz[,q_weight]");
            for(std::size_t i=0;i<expected.size();++i) {
                auto name=fields[i];
                std::transform(name.begin(),name.end(),name.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
                if(name!=expected[i]) throw std::runtime_error("CSV geometry header must be x,y,z,mx,my,mz[,q_weight]");
            }
            if(fields.size()==7) {
                auto name=fields[6];
                std::transform(name.begin(),name.end(),name.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
                if(name!="q_weight")
                    throw std::runtime_error("Seventh CSV geometry column must be q_weight");
                have_order_weights=true;
            }
            header_read=true; continue;
        }
        const auto expected_fields=have_order_weights?7U:6U;
        if(fields.size()!=expected_fields) throw std::runtime_error("CSV geometry line "+std::to_string(line_number)+" has the wrong column count");
        std::array<double,7> value{};
        for(std::size_t i=0;i<expected_fields;++i) {
            std::size_t used=0;
            try { value[i]=std::stod(fields[i],&used); }
            catch(const std::exception&) { throw std::runtime_error("Invalid number at CSV geometry line "+std::to_string(line_number)); }
            if(used!=fields[i].size() || !std::isfinite(value[i]))
                throw std::runtime_error("Invalid number at CSV geometry line "+std::to_string(line_number));
        }
        result.positions.push_back({value[0],value[1],value[2]});
        result.axes.push_back({value[3],value[4],value[5]});
        if(have_order_weights) result.order_weights.push_back(value[6]);
    }
    if(!header_read) throw std::runtime_error("CSV geometry file has no header");
    result.validate();
    return result;
}

Vec3 minimum_image(Vec3 d, const Box& box) {
    if (!box.periodic) return d;
    if (box.lengths.x > 0.0)
        d.x -= box.lengths.x * std::nearbyint(d.x / box.lengths.x);
    if (box.lengths.y > 0.0)
        d.y -= box.lengths.y * std::nearbyint(d.y / box.lengths.y);
    if (box.lengths.z > 0.0)
        d.z -= box.lengths.z * std::nearbyint(d.z / box.lengths.z);
    return d;
}

double dipolar_pair(Vec3 displacement, Vec3 axis_i, Vec3 axis_j,
                    double coupling_scale) {
    const auto r2 = dot(displacement, displacement);
    if (!(r2 > 0.0)) throw std::invalid_argument("Coincident dipole sites");
    const auto inv_r = 1.0 / std::sqrt(r2);
    const Vec3 rhat{displacement.x*inv_r, displacement.y*inv_r, displacement.z*inv_r};
    const auto angular = dot(axis_i, axis_j) - 3.0*dot(axis_i, rhat)*dot(axis_j, rhat);
    return coupling_scale * angular * inv_r * inv_r * inv_r;
}

DenseCouplings::DenseCouplings(const Geometry& geometry, double scale)
    : n_(geometry.size()), matrix_(n_*n_, 0.0) {
    geometry.validate();
    #pragma omp parallel for schedule(static)
    for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(n_); ++ii) {
        const auto i = static_cast<std::size_t>(ii);
        for (std::size_t j = i + 1; j < n_; ++j) {
            const auto value = pair_for(geometry, i, j, scale);
            matrix_[i*n_ + j] = value;
            matrix_[j*n_ + i] = value;
        }
    }
}

double DenseCouplings::at(std::size_t i, std::size_t j) const noexcept {
    return matrix_[i*n_ + j];
}

void DenseCouplings::compute_fields(std::span<const std::int8_t> spins,
                                    std::span<double> fields) const {
    if(spins.size()!=n_||fields.size()!=n_)
        throw std::invalid_argument("Dense field size mismatch");
    #pragma omp parallel for schedule(static)
    for(std::int64_t ii=0;ii<static_cast<std::int64_t>(n_);++ii) {
        const auto i=static_cast<std::size_t>(ii);
        const auto* row=matrix_.data()+i*n_;
        double sum=0.0;
        for(std::size_t j=0;j<n_;++j)
            sum+=row[j]*static_cast<double>(spins[j]);
        fields[i]=sum;
    }
}

void DenseCouplings::add_flip_delta(std::size_t flipped, std::int8_t old_spin,
                                    std::span<double> fields) const {
    const auto multiplier = -2.0 * static_cast<double>(old_spin);
    const auto* row=matrix_.data()+flipped*n_;
    for (std::size_t j = 0; j < n_; ++j)
        fields[j] += multiplier * row[j];
}

CsrCouplings::CsrCouplings(const Geometry& geometry, double scale, double cutoff)
    : n_(geometry.size()), offsets_(n_ + 1, 0) {
    geometry.validate();
    if (!(cutoff > 0.0)) throw std::invalid_argument("CSR cutoff must be positive");
    struct Edge { std::size_t row, column; double value; };
    std::vector<Edge> edges;
    for (std::size_t i = 0; i < n_; ++i) {
        for (std::size_t j = i + 1; j < n_; ++j) {
            auto d = minimum_image(subtract(geometry.positions[j], geometry.positions[i]), geometry.box);
            if (norm(d) <= cutoff) {
                const auto value = dipolar_pair(d, geometry.axes[i], geometry.axes[j], scale);
                edges.push_back({i, j, value});
                edges.push_back({j, i, value});
            }
        }
    }
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return std::tie(a.row, a.column) < std::tie(b.row, b.column);
    });
    for (const auto& edge : edges) ++offsets_[edge.row + 1];
    for (std::size_t i = 1; i < offsets_.size(); ++i) offsets_[i] += offsets_[i-1];
    neighbors_.reserve(edges.size()); values_.reserve(edges.size());
    for (const auto& edge : edges) {
        neighbors_.push_back(edge.column);
        values_.push_back(edge.value);
    }
}

double CsrCouplings::at(std::size_t i, std::size_t j) const noexcept {
    for (auto k = offsets_[i]; k < offsets_[i+1]; ++k)
        if (neighbors_[k] == j) return values_[k];
    return 0.0;
}

void CsrCouplings::compute_fields(std::span<const std::int8_t> spins,
                                  std::span<double> fields) const {
    if(spins.size()!=n_||fields.size()!=n_)
        throw std::invalid_argument("CSR field size mismatch");
    #pragma omp parallel for schedule(static)
    for(std::int64_t ii=0;ii<static_cast<std::int64_t>(n_);++ii) {
        const auto i=static_cast<std::size_t>(ii);
        double sum=0.0;
        for(auto k=offsets_[i];k<offsets_[i+1];++k)
            sum+=values_[k]*static_cast<double>(spins[neighbors_[k]]);
        fields[i]=sum;
    }
}

void CsrCouplings::add_flip_delta(std::size_t flipped, std::int8_t old_spin,
                                  std::span<double> fields) const {
    const auto multiplier = -2.0 * static_cast<double>(old_spin);
    for (auto k = offsets_[flipped]; k < offsets_[flipped+1]; ++k)
        fields[neighbors_[k]] += multiplier * values_[k];
}

std::vector<double> local_fields(const Couplings& couplings,
                                 std::span<const std::int8_t> spins) {
    if (couplings.size() != spins.size()) throw std::invalid_argument("Spin count mismatch");
    std::vector<double> result(spins.size(), 0.0);
    couplings.compute_fields(spins,result);
    return result;
}

double total_energy(const Couplings& couplings, std::span<const std::int8_t> spins) {
    const auto fields = local_fields(couplings, spins);
    return energy_from_fields(spins,fields);
}

double energy_from_fields(std::span<const std::int8_t> spins,
                          std::span<const double> fields) {
    if(spins.size()!=fields.size()) throw std::invalid_argument("Spin/field size mismatch");
    double energy = 0.0;
    for (std::size_t i = 0; i < spins.size(); ++i)
        energy += 0.5 * static_cast<double>(spins[i]) * fields[i];
    return energy;
}

double flip_delta(std::size_t i, std::span<const std::int8_t> spins,
                  std::span<const double> fields) {
    if (i >= spins.size() || spins.size() != fields.size())
        throw std::out_of_range("Invalid spin/field index");
    return -2.0 * static_cast<double>(spins[i]) * fields[i];
}

} // namespace wl
