#include "rascaline.hpp"
#include "catch.hpp"

#include "test_system.hpp"

const char* HYPERS_JSON = R"({
    "cutoff": 3.0,
    "delta": 5,
    "name": "bar",
    "gradients": true
})";

static void compute_descriptor(rascaline::Descriptor& descriptor) {
    auto calculator = rascaline::Calculator("dummy_calculator", HYPERS_JSON);

    auto system = TestSystem();
    auto systems = std::vector<rascaline::System*>();
    systems.push_back(&system);

    calculator.compute(std::move(systems), descriptor);
}

TEST_CASE("Descriptor") {
    SECTION("features") {
        auto descriptor = rascaline::Descriptor();

        auto features = descriptor.features();
        CHECK(features.names().size() == 0);
        CHECK(features.shape() == std::array<size_t, 2>{0, 0});
        CHECK(features.is_empty());

        compute_descriptor(descriptor);

        features = descriptor.features();
        CHECK(features.names()[0] == "index_delta");
        CHECK(features.names()[1] == "x_y_z");
        CHECK(features.shape() == std::array<size_t, 2>{2, 2});

        CHECK(features(0, 0) == 1);
        CHECK(features(0, 1) == 0);
        CHECK(features(1, 0) == 0);
        CHECK(features(1, 1) == 1);

        CHECK(features.position({1, 0}) == 0);
        CHECK(features.position({0, 1}) == 1);
        CHECK(features.position({2, 1}) == RASCAL_NOT_FOUND);
    }

    SECTION("samples") {
        auto descriptor = rascaline::Descriptor();

        auto samples = descriptor.samples();
        CHECK(samples.names().size() == 0);
        CHECK(samples.shape() == std::array<size_t, 2>{0, 0});
        CHECK(samples.is_empty());

        compute_descriptor(descriptor);

        samples = descriptor.samples();
        CHECK(samples.names()[0] == "structure");
        CHECK(samples.names()[1] == "center");
        CHECK(samples.shape() == std::array<size_t, 2>{4, 2});

        for (size_t i=0; i<samples.shape()[0]; i++) {
            // structure 0, atom i
            CHECK(samples(i, 0) == 0);
            CHECK(samples(i, 1) == i);
        }

        CHECK(samples.position({0, 0}) == 0);
        CHECK(samples.position({0, 3}) == 3);
        CHECK(samples.position({1, 3}) == RASCAL_NOT_FOUND);
    }

    SECTION("values") {
        auto descriptor = rascaline::Descriptor();

        auto values = descriptor.values();
        CHECK(values.shape() == std::array<size_t, 2>{0, 0});
        CHECK(values.is_empty());

        compute_descriptor(descriptor);
        values = descriptor.values();
        CHECK(values.shape() == std::array<size_t, 2>{4, 2});

        CHECK(values(0, 0) == 5);
        CHECK(values(0, 1) == 3);
        CHECK(values(1, 0) == 6);
        CHECK(values(1, 1) == 9);
        CHECK(values(2, 0) == 7);
        CHECK(values(2, 1) == 18);
        CHECK(values(3, 0) == 8);
        CHECK(values(3, 1) == 15);
    }

    SECTION("gradient samples") {
        auto descriptor = rascaline::Descriptor();

        auto gradients_samples = descriptor.gradients_samples();
        CHECK(gradients_samples.names().size() == 0);
        CHECK(gradients_samples.shape() == std::array<size_t, 2>{0, 0});
        CHECK(gradients_samples.is_empty());

        compute_descriptor(descriptor);
        gradients_samples = descriptor.gradients_samples();
        CHECK(gradients_samples.shape() == std::array<size_t, 2>{18, 4});

        auto expected = std::vector<int32_t> {
            // structure, atom, neighbor atom, spatial
            /* x */ 0, 0, 1, 0, /* y */ 0, 0, 1, 1, /* z */ 0, 0, 1, 2,
            /* x */ 0, 1, 0, 0, /* y */ 0, 1, 0, 1, /* z */ 0, 1, 0, 2,
            /* x */ 0, 1, 2, 0, /* y */ 0, 1, 2, 1, /* z */ 0, 1, 2, 2,
            /* x */ 0, 2, 1, 0, /* y */ 0, 2, 1, 1, /* z */ 0, 2, 1, 2,
            /* x */ 0, 2, 3, 0, /* y */ 0, 2, 3, 1, /* z */ 0, 2, 3, 2,
            /* x */ 0, 3, 2, 0, /* y */ 0, 3, 2, 1, /* z */ 0, 3, 2, 2,
        };

        auto count = gradients_samples.shape()[0];
        auto size = gradients_samples.shape()[1];
        for (size_t i=0; i<count; i++) {
            for (size_t j=0; j<size; j++) {
                CHECK(gradients_samples(i, j) == expected[i * size + j]);
            }
        }

        CHECK(gradients_samples.position({0, 1, 2, 1}) == 7);
        CHECK(gradients_samples.position({0, 2, 3, 2}) == 14);
        CHECK(gradients_samples.position({0, 2, 0, 0}) == RASCAL_NOT_FOUND);

        CHECK(gradients_samples.names()[0] == "structure");
        CHECK(gradients_samples.names()[1] == "center");
        CHECK(gradients_samples.names()[2] == "neighbor");
        CHECK(gradients_samples.names()[3] == "spatial");
    }

    SECTION("gradients") {
        auto descriptor = rascaline::Descriptor();

        auto gradients = descriptor.gradients();
        CHECK(gradients.shape() == std::array<size_t, 2>{0, 0});
        CHECK(gradients.is_empty());

        compute_descriptor(descriptor);
        gradients = descriptor.gradients();
        CHECK(gradients.shape() == std::array<size_t, 2>{18, 2});

        for (size_t i=0; i<gradients.shape()[0]; i++) {
            CHECK(gradients(i, 0) == 0);
            CHECK(gradients(i, 1) == 1);
        }
    }

    SECTION("densify") {
        auto descriptor = rascaline::Descriptor();
        compute_descriptor(descriptor);

        CHECK(descriptor.values().shape() == std::array<size_t, 2>{4, 2});

        descriptor.densify({"center"});
        CHECK(descriptor.values().shape() == std::array<size_t, 2>{1, 8});

        CHECK_THROWS_WITH(
            descriptor.densify({"not there"}),
            "internal error: can not densify along 'not there' which is not "
            "present in the samples: [structure]"
        );
    }
}
