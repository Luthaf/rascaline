#include "rascaline.hpp"
#include "catch.hpp"


TEST_CASE("basic systems") {
    auto systems = rascaline::BasicSystems(
        "../../../rascaline/benches/data/silicon_bulk.xyz"
    );

    CHECK(systems.count() == 30);

    auto* system = systems.systems();
    uintptr_t size = 0;
    system->size(system->user_data, &size);
    CHECK(size == 54);

    const int32_t* types = nullptr;
    system->types(system->user_data, &types);
    for (size_t i=0; i<size; i++) {
        CHECK(types[i] == 14);
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
        throw std::runtime_error("unimplemented function 'size'");
    }

    const int32_t* types() const override {
        throw std::runtime_error("unimplemented function 'types'");
    }

    const double* positions() const override {
        throw std::runtime_error("unimplemented function 'positions'");
    }

    CellMatrix cell() const override {
        throw std::runtime_error("unimplemented function 'cell'");
    }

    void compute_neighbors(double cutoff) override {
        throw std::runtime_error("unimplemented function 'compute_neighbors'");
    }

    const std::vector<rascal_pair_t>& pairs() const override {
        throw std::runtime_error("unimplemented function 'pairs'");
    }

    const std::vector<rascal_pair_t>& pairs_containing(uintptr_t atom) const override {
        throw std::runtime_error("unimplemented function 'pairs_containing'");
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
    auto calculator = rascaline::Calculator("dummy_calculator", HYPERS_JSON);

    CHECK_THROWS_WITH(calculator.compute(system), "unimplemented function 'types'");
}
