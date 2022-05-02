#include <stdbool.h>
#include <stdio.h>

#include <rascaline.h>

/// Compute SOAP power spectrum, this is the same code as the 'compute-soap'
/// example
static rascal_status_t compute_soap(rascal_descriptor_t* descriptor, const char* path);

int main(int argc, char* argv[]) {
    rascal_status_t status = RASCAL_SUCCESS;
    rascal_descriptor_t* descriptor = NULL;
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

    descriptor = rascal_descriptor();
    if (descriptor == NULL) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    status = compute_soap(descriptor, argv[1]);
    if (status != RASCAL_SUCCESS) {
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
    rascal_descriptor_free(descriptor);

    if (got_error) {
        return 1;
    } else {
        return 0;
    }
}

int compute_soap(rascal_descriptor_t* descriptor, const char* path) {
    rascal_status_t status = RASCAL_SUCCESS;
    rascal_calculator_t* calculator;
    rascal_system_t* systems = NULL;
    uintptr_t n_systems = 0;
    double* values = NULL;
    uintptr_t n_samples = 0;
    uintptr_t n_features = 0;
    const char* densify_variables[] = {"species_neighbor_1", "species_neighbor_2"};
    rascal_calculation_options_t options = {0};

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
        calculator, descriptor, systems, n_systems, options
    );
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    status = rascal_descriptor_densify(descriptor, densify_variables, 1, NULL, 0);
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    status = rascal_descriptor_values(descriptor, &values, &n_samples, &n_features);
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

cleanup:
    rascal_calculator_free(calculator);
    rascal_basic_systems_free(systems, n_systems);

    return status;
}
