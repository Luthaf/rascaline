#include "featomic.h"
#include "catch.hpp"


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
