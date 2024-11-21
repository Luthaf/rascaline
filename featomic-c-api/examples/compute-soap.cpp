#include <iostream>
#include <chemfiles.hpp>
#include <featomic.hpp>

static std::vector<featomic::SimpleSystem> read_systems(const std::string& path);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "error: expected a command line argument" << std::endl;
        return 1;
    }
    auto systems = read_systems(argv[1]);

    // pass hyper-parameters as JSON
    const char* parameters = R"({
        "cutoff": {
            "radius": 5.0,
            "smoothing": {"type": "ShiftedCosine", "width": 0.5}
        },
        "density": {
            "type": "Gaussian",
            "width": 0.3
        },
        "basis": {
            "type": "TensorProduct",
            "max_angular": 6,
            "radial": {"type": "Gto", "max_radial": 6}
        }
    })";

    // create the calculator with its name and parameters
    auto calculator = featomic::Calculator("soap_power_spectrum", parameters);

    // setup the calculation options
    auto options = featomic::CalculationOptions();
    options.use_native_system = true;
    options.gradients.push_back("positions");

    // run the calculation
    auto descriptor = calculator.compute(systems, options);

    // The descriptor is a metatensor `TensorMap`, containing multiple blocks.
    // We can transform it to a single block containing a dense representation,
    // with one sample for each atom-centered environment.
    descriptor.keys_to_samples("center_type");
    descriptor.keys_to_properties(std::vector<std::string>{"neighbor_1_type", "neighbor_2_type"});

    // extract values from the descriptor in the only remaining block
    auto block = descriptor.block_by_id(0);
    auto values = block.values();

    // you can now use values as the input of a machine learning algorithm

    return 0;
}


std::vector<featomic::SimpleSystem> read_systems(const std::string& path) {
    auto trajectory = chemfiles::Trajectory(path);
    auto systems = std::vector<featomic::SimpleSystem>();
    for (size_t step=0; step<trajectory.nsteps(); step++) {
        auto frame = trajectory.read_step(step);

        auto matrix = frame.cell().matrix();
        auto system = featomic::SimpleSystem(featomic::System::CellMatrix{{
            {matrix[0][0], matrix[0][1], matrix[0][2]},
            {matrix[1][0], matrix[1][1], matrix[1][2]},
            {matrix[2][0], matrix[2][1], matrix[2][2]},
        }});

        auto positions = frame.positions();
        for (size_t i=0; i<frame.size(); i++) {
            system.add_atom(
                static_cast<int32_t>(frame[i].atomic_number().value_or(0)),
                {positions[i][0], positions[i][1], positions[i][2]}
            );
        }

        systems.push_back(system);
    }

    return systems;
}
