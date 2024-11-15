#include <torch/torch.h>

#include <metatensor/torch/atomistic.hpp>
#include <featomic/torch.hpp>

using namespace featomic_torch;

int main() {
    auto system = torch::make_intrusive<metatensor_torch::SystemHolder>(
        torch::zeros({5}, torch::kI32),
        torch::rand({5, 3}, torch::kF64),
        torch::zeros({3, 3}, torch::kF64),
        torch::zeros({3}, torch::kBool)
    );

    const auto* HYPERS_JSON = R"({
        "cutoff": 3.0,
        "delta": 4,
        "name": "bar"
    })";
    auto calculator = CalculatorHolder("dummy_calculator", HYPERS_JSON);

    auto descriptor = calculator.compute({system});

    return 0;
}
