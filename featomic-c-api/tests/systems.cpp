#include "featomic.h"
#include "catch.hpp"
#include "helpers.hpp"


TEST_CASE("basic systems") {
    featomic_system_t* systems = nullptr;
    uintptr_t count = 0;

    const char* path = "../../featomic/benches/data/silicon_bulk.xyz";
    CHECK_SUCCESS(featomic_basic_systems_read(path, &systems, &count));
    CHECK(count == 30);

    auto system = systems[0];

    uintptr_t size = 0;
    system.size(system.user_data, &size);
    CHECK(size == 54);

    const int32_t* types = nullptr;
    system.types(system.user_data, &types);
    for (size_t i=0; i<size; i++) {
        CHECK(types[i] == 14);
    }

    const double* positions = nullptr;
    system.positions(system.user_data, &positions);
    CHECK_THAT(positions[0], Catch::Matchers::WithinULP(7.8554, 10));
    CHECK_THAT(positions[1], Catch::Matchers::WithinULP(7.84887, 10));
    CHECK_THAT(positions[2], Catch::Matchers::WithinULP(0.0188612, 10));

    double cell[9] = {0.0};
    system.cell(system.user_data, cell);
    CHECK_THAT(cell[0], Catch::Matchers::WithinULP(7.84785, 10));
    CHECK_THAT(cell[1], Catch::Matchers::WithinULP(0.0, 10));
    CHECK_THAT(cell[2], Catch::Matchers::WithinULP(7.84785, 10));

    CHECK_THAT(cell[3], Catch::Matchers::WithinULP(7.84785, 10));
    CHECK_THAT(cell[4], Catch::Matchers::WithinULP(7.84785, 10));
    CHECK_THAT(cell[5], Catch::Matchers::WithinULP(0.0, 10));

    CHECK_THAT(cell[6], Catch::Matchers::WithinULP(0.0, 10));
    CHECK_THAT(cell[7], Catch::Matchers::WithinULP(7.84785, 10));
    CHECK_THAT(cell[8], Catch::Matchers::WithinULP(7.84785, 10));

    CHECK_SUCCESS(featomic_basic_systems_free(systems, count));
}


TEST_CASE("systems errors") {
    const char* HYPERS_JSON = R"({
        "cutoff": 3.0,
        "delta": 4,
        "name": ""
    })";

    auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
    REQUIRE(calculator != nullptr);

    featomic_system_t system = {};
    featomic_calculation_options_t options = {};

    mts_tensormap_t* descriptor = nullptr;
    auto status = featomic_calculator_compute(
        calculator, &descriptor, &system, 1, options
    );
    CHECK(descriptor == nullptr);
    CHECK(status == FEATOMIC_SYSTEM_ERROR);

    std::string expected = "error from external code (status 128): featomic_system_t.types function is NULL";
    CHECK(featomic_last_error() == expected);

    system.types = [](const void* _, const int32_t** types) {
        return -5242832;
    };

    status = featomic_calculator_compute(
        calculator, &descriptor, &system, 1, options
    );
    CHECK(descriptor == nullptr);
    CHECK(status == -5242832);
    expected = "error from external code (status -5242832): call to featomic_system_t.types failed";
    CHECK(featomic_last_error() == expected);

    featomic_calculator_free(calculator);
    mts_tensormap_free(descriptor);
}
