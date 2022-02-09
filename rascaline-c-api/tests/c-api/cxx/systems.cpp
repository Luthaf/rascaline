#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "rascaline.hpp"


TEST_CASE("basic systems") {
    auto systems = rascaline::BasicSystems(
        "../../../../rascaline/benches/data/silicon_bulk.xyz"
    );

    CHECK(systems.count() == 30);

    auto system = systems.systems();
    uintptr_t size = 0;
    system->size(system->user_data, &size);
    CHECK(size == 54);

    const int32_t* species = nullptr;
    system->species(system->user_data, &species);
    for (size_t i=0; i<size; i++) {
        CHECK(species[i] == 14);
    }

    const double* positions = nullptr;
    system->positions(system->user_data, &positions);
    CHECK_THAT(positions[0], Catch::Matchers::WithinULP(7.8554, 10));
    CHECK_THAT(positions[1], Catch::Matchers::WithinULP(7.84887, 10));
    CHECK_THAT(positions[2], Catch::Matchers::WithinULP(0.0188612, 10));

    double cell[9] = {0.0};
    system->cell(system->user_data, cell);
    CHECK_THAT(cell[0], Catch::Matchers::WithinULP(7.84785, 10));
    CHECK_THAT(cell[1], Catch::Matchers::WithinULP(0.0, 10));
    CHECK_THAT(cell[2], Catch::Matchers::WithinULP(7.84785, 10));

    CHECK_THAT(cell[3], Catch::Matchers::WithinULP(7.84785, 10));
    CHECK_THAT(cell[4], Catch::Matchers::WithinULP(7.84785, 10));
    CHECK_THAT(cell[5], Catch::Matchers::WithinULP(0.0, 10));

    CHECK_THAT(cell[6], Catch::Matchers::WithinULP(0.0, 10));
    CHECK_THAT(cell[7], Catch::Matchers::WithinULP(7.84785, 10));
    CHECK_THAT(cell[8], Catch::Matchers::WithinULP(7.84785, 10));
}

class BadSystem: public rascaline::System {
public:
    uintptr_t size() const override {
        throw std::runtime_error("this is a test error");
    }

    const int32_t* species() const override {
        throw std::runtime_error("unimplemented");
    }

    const double* positions() const override {
        throw std::runtime_error("unimplemented");
    }

    CellMatrix cell() const override {
        throw std::runtime_error("unimplemented");
    }

    void compute_neighbors(double cutoff) override {
        throw std::runtime_error("unimplemented");
    }

    const std::vector<rascal_pair_t>& pairs() const override {
        throw std::runtime_error("unimplemented");
    }

    const std::vector<rascal_pair_t>& pairs_containing(uintptr_t center) const override {
        throw std::runtime_error("unimplemented");
    }
};

TEST_CASE("systems errors") {
    const char* HYPERS_JSON = R"({
        "cutoff": 3.0,
        "delta": 4,
        "name": "",
        "gradients": true
    })";

    auto system = BadSystem();
    auto systems = std::vector<rascaline::System*>();
    systems.push_back(&system);
    auto calculator = rascaline::Calculator("dummy_calculator", HYPERS_JSON);

    CHECK_THROWS_WITH(calculator.compute(systems), "this is a test error");
}
