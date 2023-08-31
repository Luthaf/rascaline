#include "rascaline.h"
#include "catch.hpp"
#include "helpers.hpp"


TEST_CASE("basic systems") {
    rascal_system_t* systems = nullptr;
    uintptr_t count = 0;

    const char* path = "../../rascaline/benches/data/silicon_bulk.xyz";
    CHECK_SUCCESS(rascal_basic_systems_read(path, &systems, &count));
    CHECK(count == 30);

    auto system = systems[0];

    uintptr_t size = 0;
    system.size(system.user_data, &size);
    CHECK(size == 54);

    const int32_t* species = nullptr;
    system.species(system.user_data, &species);
    for (size_t i=0; i<size; i++) {
        CHECK(species[i] == 14);
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

    CHECK_SUCCESS(rascal_basic_systems_free(systems, count));
}


TEST_CASE("systems errors") {
    const char* HYPERS_JSON = R"({
        "cutoff": 3.0,
        "delta": 4,
        "name": ""
    })";

    auto* calculator = rascal_calculator("dummy_calculator", HYPERS_JSON);
    REQUIRE(calculator != nullptr);

    rascal_system_t system = {0};
    rascal_calculation_options_t options = {0};

    mts_tensormap_t* descriptor = nullptr;
    auto status = rascal_calculator_compute(
        calculator, &descriptor, &system, 1, options
    );
    CHECK(descriptor == nullptr);
    CHECK(status == RASCAL_SYSTEM_ERROR);

    std::string expected = "error from external code (status 128): rascal_system_t.species function is NULL";
    CHECK(rascal_last_error() == expected);

    system.species = [](const void* _, const int32_t** species) {
        return -5242832;
    };

    status = rascal_calculator_compute(
        calculator, &descriptor, &system, 1, options
    );
    CHECK(descriptor == nullptr);
    CHECK(status == -5242832);
    expected = "error from external code (status -5242832): call to rascal_system_t.species failed";
    CHECK(rascal_last_error() == expected);

    rascal_calculator_free(calculator);
    mts_tensormap_free(descriptor);
}
