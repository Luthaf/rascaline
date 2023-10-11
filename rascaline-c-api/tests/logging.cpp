#include <vector>
#include <string>

#include "catch.hpp"
#include "helpers.hpp"

#include "metatensor.h"
#include "rascaline.h"

static std::vector<std::tuple<int, std::string>> RECORDED_LOG_EVENTS;

static void run_calculation(const char* hypers) {
    auto* calculator = rascal_calculator("dummy_calculator", hypers);
    REQUIRE(calculator != nullptr);
    auto system = simple_system();
    rascal_calculation_options_t options = {};

    mts_tensormap_t* descriptor = nullptr;
    CHECK_SUCCESS(rascal_calculator_compute(
        calculator, &descriptor, &system, 1, options
    ));

    rascal_calculator_free(calculator);
    mts_tensormap_free(descriptor);
}

TEST_CASE("Logging") {
    auto record_log_events = [](int level, const char* message) {
        RECORDED_LOG_EVENTS.emplace_back(level, std::string(message));
    };
    CHECK_SUCCESS(rascal_set_logging_callback(record_log_events));

    const char* hypers_log_info = R"({
        "cutoff": 3.0,
        "delta": 0,
        "name": "log-test-info: test info message",
        "gradients": false
    })";

    RECORDED_LOG_EVENTS.clear();
    run_calculation(hypers_log_info);

    bool event_found = false;
    for (const auto& event: RECORDED_LOG_EVENTS) {
        if (std::get<1>(event) == "rascaline::calculators::dummy_calculator -- log-test-info: test info message") {
            CHECK(std::get<0>(event) == RASCAL_LOG_LEVEL_INFO);
            event_found = true;
        }
    }
    CHECK(event_found);


    const char* hypers_log_warn = R"({
        "cutoff": 3.0,
        "delta": 0,
        "name": "log-test-warn: test warning message",
        "gradients": false
    })";

    RECORDED_LOG_EVENTS.clear();
    run_calculation(hypers_log_warn);

    event_found = false;
    for (const auto& event: RECORDED_LOG_EVENTS) {
        if (std::get<1>(event) == "rascaline::calculators::dummy_calculator -- log-test-warn: test warning message") {
            CHECK(std::get<0>(event) == RASCAL_LOG_LEVEL_WARN);
            event_found = true;
        }
    }
    CHECK(event_found);
}
