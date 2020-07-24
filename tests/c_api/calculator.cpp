#include <string>

#include "rascaline.h"
#include "catch.hpp"
#include "helpers.hpp"

TEST_CASE("calculator name") {
    const char *HYPERS_JSON = R"({
    "cutoff": 3.5,
    "delta": 25,
    "name": "bar",
    "gradients": false
    })";
    auto* calculator = rascal_calculator("dummy_calculator", HYPERS_JSON);
    REQUIRE(calculator != nullptr);

    char buffer[256] = {0};
    CHECK_SUCCESS(rascal_calculator_name(calculator, buffer, sizeof(buffer)));
    CHECK(buffer == std::string("dummy test calculator with cutoff: 3.5 - delta: 25 - name: bar - gradients: false"));

    rascal_calculator_free(calculator);
}

TEST_CASE("calculator creation error") {
    const char *HYPERS_JSON = R"({
    "cutoff": "22",
    "delta": 25,
    "name": "bar",
    "gradients": false
    })";
    auto *calculator = rascal_calculator("dummy_calculator", HYPERS_JSON);
    CHECK(calculator == nullptr);

    CHECK(rascal_last_error() == std::string("json error: invalid type: string \"22\", expected f64 at line 2 column 18"));
}
