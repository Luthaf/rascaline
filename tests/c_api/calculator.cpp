#include <string>

#include "rascaline.h"
#include "catch.hpp"
#include "helpers.hpp"

TEST_CASE("calculator name") {
    SECTION("dummy_calculator") {
        const char* HYPERS_JSON = R"({
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

    SECTION("long strings") {
        auto name = std::string(2048, 'b');
        auto HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "gradients": false,
            "name": ")" + name + "\"}";

        auto* calculator = rascal_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        char* buffer = new char[4096];
        auto status = rascal_calculator_name(calculator, buffer, 256);
        CHECK(status == RASCAL_INVALID_PARAMETER_ERROR);

        CHECK_SUCCESS(rascal_calculator_name(calculator, buffer, 4096));
        std::string expected = "dummy test calculator with cutoff: 3.5 - delta: 25 - ";
        expected += "name: " + name + " - gradients: false";
        CHECK(buffer == expected);

        delete[] buffer;

        rascal_calculator_free(calculator);
    }
}

TEST_CASE("calculator parameters") {
    SECTION("dummy_calculator") {
        std::string HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": "bar",
            "gradients": false
        })";
        auto* calculator = rascal_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        char buffer[256] = {0};
        CHECK_SUCCESS(rascal_calculator_parameters(calculator, buffer, sizeof(buffer)));
        CHECK(buffer == HYPERS_JSON);

        rascal_calculator_free(calculator);
    }

    SECTION("long strings") {
        auto name = std::string(2048, 'b');
        auto HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "gradients": false,
            "name": ")" + name + "\"}";

        auto* calculator = rascal_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        char* buffer = new char[4096];
        auto status = rascal_calculator_parameters(calculator, buffer, 256);
        CHECK(status == RASCAL_INVALID_PARAMETER_ERROR);

        CHECK_SUCCESS(rascal_calculator_parameters(calculator, buffer, 4096));
        CHECK(buffer == HYPERS_JSON);

        delete[] buffer;

        rascal_calculator_free(calculator);
    }
}

TEST_CASE("calculator creation error") {
    const char* HYPERS_JSON = R"({
        "cutoff": "532",
        "delta": 25,
        "name": "bar",
        "gradients": false
    })";
    auto *calculator = rascal_calculator("dummy_calculator", HYPERS_JSON);
    CHECK(calculator == nullptr);

    CHECK(rascal_last_error() == std::string("json error: invalid type: string \"532\", expected f64 at line 2 column 23"));
}
