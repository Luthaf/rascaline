#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include <rascaline.h>
#include <equistore.h>

int main(int argc, char* argv[]) {
    int status = RASCAL_SUCCESS;
    rascal_calculator_t* calculator = NULL;
    rascal_system_t* systems = NULL;
    uintptr_t n_systems = 0;
    const double* values = NULL;
    const uintptr_t* shape = NULL;
    uintptr_t shape_count = 0;
    bool got_error = true;
    const char* keys_to_samples[] = {"species_center"};
    const char* keys_to_properties[] = {"species_neighbor_1", "species_neighbor_2"};
    // use the default set of options, computing all samples and all features
    rascal_calculation_options_t options = {0};

    eqs_tensormap_t* descriptor = NULL;
    const eqs_block_t* block = NULL;
    eqs_array_t data = {0};
    eqs_labels_t keys_to_move = {0};

    // hyper-parameters for the calculation as JSON
    const char* parameters = "{\n"
        "\"cutoff\": 5.0,\n"
        "\"max_radial\": 6,\n"
        "\"max_angular\": 4,\n"
        "\"atomic_gaussian_width\": 0.3,\n"
        "\"center_atom_weight\": 1.0,\n"
        "\"gradients\": false,\n"
        "\"radial_basis\": {\n"
        "    \"Gto\": {}\n"
        "},\n"
        "\"cutoff_function\": {\n"
        "    \"ShiftedCosine\": {\"width\": 0.5}\n"
        "}\n"
    "}";

    // load systems from command line arguments
    if (argc < 2) {
        printf("error: expected a command line argument");
        goto cleanup;
    }
    status = rascal_basic_systems_read(argv[1], &systems, &n_systems);
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    // create the calculator with its name and parameters
    calculator = rascal_calculator("soap_power_spectrum", parameters);
    if (calculator == NULL) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    // run the calculation
    status = rascal_calculator_compute(
        calculator, &descriptor, systems, n_systems, options
    );
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    // The descriptor is an equistore `TensorMap`, containing multiple blocks.
    // We can transform it to a single block containing a dense representation,
    // with one sample for each atom-centered environment.
    keys_to_move.names = keys_to_samples;
    keys_to_move.size = 1;
    keys_to_move.values = NULL;
    keys_to_move.count = 0;
    status = eqs_tensormap_keys_to_samples(descriptor, keys_to_move, true);
    if (status != EQS_SUCCESS) {
        printf("Error: %s\n", eqs_last_error());
        goto cleanup;
    }

    keys_to_move.names = keys_to_properties;
    keys_to_move.size = 2;
    keys_to_move.values = NULL;
    keys_to_move.count = 0;
    status = eqs_tensormap_keys_to_properties(descriptor, keys_to_move, true);
    if (status != EQS_SUCCESS) {
        printf("Error: %s\n", eqs_last_error());
        goto cleanup;
    }

    // extract the unique block and corresponding values from the descriptor
    status = eqs_tensormap_block_by_id(descriptor, &block, 0);
    if (status != EQS_SUCCESS) {
        printf("Error: %s\n", eqs_last_error());
        goto cleanup;
    }

    status = eqs_block_data(block, "values", &data);
    if (status != EQS_SUCCESS) {
        printf("Error: %s\n", eqs_last_error());
        goto cleanup;
    }

    status = eqs_get_rust_array(&data, &values, &shape, &shape_count);
    if (status != EQS_SUCCESS) {
        printf("Error: %s\n", eqs_last_error());
        goto cleanup;
    }
    assert(shape_count == 2);

    // you can now use `values` as the input of a machine learning algorithm
    printf("the value array shape is %lu x %lu\n", shape[0], shape[1]);

    got_error = false;
cleanup:
    eqs_tensormap_free(descriptor);
    rascal_calculator_free(calculator);
    rascal_basic_systems_free(systems, n_systems);

    if (got_error) {
        return 1;
    } else {
        return 0;
    }
}
