#include <torch/torch.h>

#include <rascaline/torch.hpp>
using namespace rascaline_torch;

#include <catch.hpp>

TEST_CASE("Systems") {
    SECTION("Basic usage") {
        auto positions = torch::zeros({5, 3});
        auto species = torch::zeros({5}, torch::TensorOptions(torch::kInt32));
        auto cell = torch::zeros({3, 3});
        auto system = SystemHolder(species, positions, cell);

        CHECK(system.size() == 5);
        CHECK(system.__len__() == 5);

        CHECK(system.use_native_system() == true);

        system.set_precomputed_pairs(3.2, {{0, 1, 0.0, {0.0, 0.0, 0.0}, {0, 1, 0}}});
        system.set_precomputed_pairs(4.5, {{3, 2, 0.0, {0.0, 0.0, 0.0}, {0, 1, 0}}});
        CHECK(system.use_native_system() == false);

        CHECK_THROWS_WITH(
            system.compute_neighbors(3.3),
            Catch::Matchers::StartsWith(
                "trying to get neighbor list with a cutoff (3.3) for which no "
                "pre-computed neighbor lists has been registered (we have lists "
                "for cutoff=[3.2, 4.5])"
            )
        );

        system.compute_neighbors(3.2);
        REQUIRE(system.pairs().size() == 1);
        CHECK(system.pairs()[0].first == 0);
        CHECK(system.pairs()[0].second == 1);

        system.compute_neighbors(4.5);
        REQUIRE(system.pairs().size() == 1);
        CHECK(system.pairs()[0].first == 3);
        CHECK(system.pairs()[0].second == 2);
    }

    SECTION("Printing") {
        auto positions = torch::zeros({5, 3});
        auto species = torch::zeros({5}, torch::TensorOptions(torch::kInt32));
        auto cell = torch::zeros({3, 3});
        auto system = SystemHolder(species, positions, cell);

        CHECK(system.__str__() == "System with 5 atoms, non periodic");

        cell = torch::eye(3);
        system = SystemHolder(species, positions, cell);
        CHECK(system.__str__() == "System with 5 atoms, periodic cell: [1, 0, 0, 0, 1, 0, 0, 0, 1]");
    }
}
