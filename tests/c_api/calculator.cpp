#include <string>

#include "rascaline.h"
#include "catch.hpp"

const char* HYPERS_JSON = R"({
    "cutoff": 3.5,
    "delta": 25,
    "name": "bar",
    "gradients": false
})";

static rascal_system_t simple_system();


TEST_CASE("calculator name") {
    auto* calculator = rascal_calculator("dummy_calculator", HYPERS_JSON);
    REQUIRE(calculator != nullptr);

    char buffer[256] = {0};
    auto status = rascal_calculator_name(calculator, buffer, sizeof(buffer));
    CHECK(status == RASCAL_SUCCESS);
    CHECK(buffer == std::string("dummy test calculator with cutoff: 3.5 - delta: 25 - name: bar - gradients: false"));

    rascal_calculator_free(calculator);
}
