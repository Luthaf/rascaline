#include <iostream>
#include <rascaline.hpp>

/// Compute SOAP power spectrum, this is the same code as the 'compute-soap'
/// example
static metatensor::TensorMap compute_soap(const std::string& path);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "error: expected a command line argument" << std::endl;
        return 1;
    }

    // enable collection of profiling data
    rascaline::Profiler::enable(true);
    // clear any existing collected data
    rascaline::Profiler::clear();

    auto descriptor = compute_soap(argv[1]);

    // Get the profiling data as a table to display it directly
    std::cout << rascaline::Profiler::get("short_table") << std::endl;
    // Or save this data as json for future usage
    std::cout << rascaline::Profiler::get("json") << std::endl;

    return 0;
}


metatensor::TensorMap compute_soap(const std::string& path) {
    auto systems = rascaline::BasicSystems(path);

    const char* parameters = R"({
        "cutoff": 5.0,
        "max_radial": 6,
        "max_angular": 4,
        "atomic_gaussian_width": 0.3,
        "center_atom_weight": 1.0,
        "gradients": false,
        "radial_basis": {
            "Gto": {}
        },
        "cutoff_function": {
            "ShiftedCosine": {"width": 0.5}
        }
    })";

    auto calculator = rascaline::Calculator("soap_power_spectrum", parameters);
    auto descriptor = calculator.compute(systems);

    descriptor.keys_to_samples("center_type");
    descriptor.keys_to_properties(std::vector<std::string>{"neighbor_1_type", "neighbor_2_type"});

    return descriptor;
}
