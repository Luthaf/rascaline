#include <stdbool.h>
#include <stdio.h>

#include <rascaline.h>

int main(int argc, char* argv[]) {
    rascal_status_t status = RASCAL_SUCCESS;
    rascal_calculator_t* calculator = NULL;
    rascal_descriptor_t* descriptor = NULL;
    rascal_system_t* systems = NULL;
    uintptr_t n_systems = 0;
    double* values = NULL;
    uintptr_t n_samples = 0;
    uintptr_t n_features = 0;
    bool got_error = true;
    const char* densify_variables[] = {"species_neighbor_1", "species_neighbor_2"};
    // use the default set of options, computing all samples and all features
    rascal_calculation_options_t options = {0};

    // hyper-parameters for the calculation as JSON
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

    // create a new empty descriptor
    descriptor = rascal_descriptor();
    if (descriptor == NULL) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    // run the calculation
    status = rascal_calculator_compute(
        calculator, descriptor, systems, n_systems, options
    );
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    // Transform the descriptor to dense representation,
    // with one sample for each atom-centered environment
    status = rascal_descriptor_densify(descriptor, densify_variables, 1, NULL, 0);
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    // extract values from the descriptor
    status = rascal_descriptor_values(descriptor, &values, &n_samples, &n_features);
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    // you can now use `values` as the input of a machine learning algorithm
    printf("the value array shape is %lu x %lu\n", n_samples, n_features);

    got_error = false;
cleanup:
    rascal_descriptor_free(descriptor);
    rascal_calculator_free(calculator);
    rascal_basic_systems_free(systems, n_systems);

    if (got_error) {
        return 1;
    } else {
        return 0;
    }
}
