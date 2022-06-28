#include <stdbool.h>
#include <stdio.h>

#include <equistore.h>
#include <rascaline.h>

/// Compute SOAP power spectrum, this is the same code as the 'compute-soap'
/// example
static eqs_tensormap_t* compute_soap(const char* path);

int main(int argc, char* argv[]) {
    rascal_status_t status = RASCAL_SUCCESS;
    char* buffer = NULL;
    size_t buffer_size = 8192;
    bool got_error = true;

    if (argc < 2) {
        printf("error: expected a command line argument");
        goto cleanup;
    }

    // enable collection of profiling data
    status = rascal_profiling_enable(true);
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    // clear any existing collected data
    status = rascal_profiling_clear();
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    eqs_tensormap_t* descriptor = compute_soap(argv[1]);
    if (descriptor == NULL) {
        goto cleanup;
    }

    buffer = calloc(buffer_size, sizeof(char));
    if (buffer == NULL) {
        printf("Error: failed to allocate memory\n");
        goto cleanup;
    }

    // Get the profiling data as a table to display it directly
    status = rascal_profiling_get("short_table", buffer, buffer_size);
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }
    printf("%s\n", buffer);

    // Or save this data as json for future usage
    status = rascal_profiling_get("json", buffer, buffer_size);
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }
    printf("%s\n", buffer);

    got_error = false;
cleanup:
    free(buffer);
    eqs_tensormap_free(descriptor);

    if (got_error) {
        return 1;
    } else {
        return 0;
    }
}

// this is the same function as in the compute-soap.c example
eqs_tensormap_t* compute_soap(const char* path) {
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
    const char* gradients_list[] = {"positions"};
    options.gradients = gradients_list;
    options.gradients_count = 1;

    eqs_tensormap_t* descriptor = NULL;
    const eqs_block_t* block = NULL;
    eqs_array_t data = {0};
    eqs_labels_t keys_to_move = {0};

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


    status = rascal_basic_systems_read(path, &systems, &n_systems);
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    calculator = rascal_calculator("soap_power_spectrum", parameters);
    if (calculator == NULL) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    status = rascal_calculator_compute(
        calculator, &descriptor, systems, n_systems, options
    );
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

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

cleanup:
    rascal_calculator_free(calculator);
    rascal_basic_systems_free(systems, n_systems);

    return descriptor;
}
