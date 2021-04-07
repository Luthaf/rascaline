#include <iostream>
#include <rascaline.hpp>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "error: expected a command line argument" << std::endl;
        return 1;
    }
    auto systems = rascaline::BasicSystems(argv[1]);

    // pass hyper-parameters as JSON
    const char* parameters = "{\n"
        "\"cutoff\": 5.0,\n"
        "\"max_radial\": 6,\n"
        "\"max_angular\": 4,\n"
        "\"atomic_gaussian_width\": 0.3,\n"
        "\"gradients\": false,\n"
        "\"radial_basis\": {\n"
        "    \"Gto\": {}\n"
        "},\n"
        "\"cutoff_function\": {\n"
        "    \"ShiftedCosine\": {\"width\": 0.5}\n"
        "}\n"
    "}";

    // create the calculator with its name and parameters
    auto calculator = rascaline::Calculator("spherical_expansion", parameters);

    // run the calculation
    auto descriptor = calculator.compute(std::move(systems));

    // Transform the descriptor to dense representation,
    // with one sample for each atom-centered environment
    descriptor.densify({"neighbor_species"});

    // extract values from the descriptor
    auto values = descriptor.values();

    // you can now use values as the input of a machine learning algorithm

    return 0;
}
