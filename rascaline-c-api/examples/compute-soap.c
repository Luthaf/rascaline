#include <stdbool.h>
#include <stdio.h>

#include <rascaline.h>

int main(int argc, char* argv[]) {
    // TODO: systems are not yet easy to create from C,
    // this code will not run for now
    rascal_system_t systems[] = {{0}, {0}};

    // pass hyper-parameters as JSON
    const char* parameters = "{\n"
        "\"cutoff\": 5.0,\n"
        "\"max_radial\": 6,\n"
        "\"max_angular\": 4,\n"
        "\"atomic_gaussian_width\": 0.3,\n"
        "\"gradients\": false,\n"
        "\"radial_basis\": {\n"
        "    \"GTO\": {}\n"
        "},\n"
        "\"cutoff_function\": {\n"
        "    \"ShiftedCosine\": {\"width\": 0.5}\n"
        "}\n"
    "}";

    rascal_calculator_t* calculator = NULL;
    rascal_descriptor_t* descriptor = NULL;
    rascal_status_t status = RASCAL_SUCCESS;
    bool got_error = true;

    // create the calculator with its name and parameters
    calculator = rascal_calculator("spherical_expansion", parameters);
    if (calculator == NULL) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    // create the calculator with its name and parameters
     descriptor = rascal_descriptor();
    if (descriptor == NULL) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    // use the default set of options, computing all samples and all features
    rascal_calculation_options_t options = {
        /* use_native_system */ false,
        /* selected_samples */ NULL,
        /* selected_samples_count */ 0,
        /* selected_features */ NULL,
        /* selected_features_count */ 0
    };

    // run the calculation
    status = rascal_calculator_compute(
        calculator, descriptor, systems, 2, options
    );
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    // Transform the descriptor to dense representation,
    // with one sample for each atom-centered environment
    const char* variables = {"neighbor_species"};
    status = rascal_descriptor_densify(descriptor, variables, 1);
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    const double* values = NULL;
    uintptr_t samples = 0;
    uintptr_t features = 0;
    status = rascal_descriptor_values(descriptor, &values, &samples, &features);
    if (status != RASCAL_SUCCESS) {
        printf("Error: %s\n", rascal_last_error());
        goto cleanup;
    }

    // you can now use values as the input of a machine learning algorithm


    got_error = false;
cleanup:
    rascal_descriptor_free(descriptor);
    rascal_calculator_free(calculator);

    if (got_error) {
        return 1;
    } else {
        return 0;
    }
}
