#include <string>

#include "rascaline.h"
#include "catch.hpp"

const char* HYPERS_JSON = R"({
    "cutoff": 3.5,
    "max_neighbors": 25,
    "padding": 3.5
})";

TEST_CASE("calculator") {
    auto* calculator = rascal_calculator("sorted_distances", HYPERS_JSON);
    REQUIRE(calculator != nullptr);

    SECTION("name") {
        char buffer[30] = {0};
        rascal_calculator_name(calculator, buffer, sizeof(buffer));
        CHECK(buffer == std::string("sorted distances vector"));
    }

    rascal_calculator_free(calculator);
}
