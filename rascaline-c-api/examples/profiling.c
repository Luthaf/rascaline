#include <stdbool.h>
#include <stdio.h>

#include <metatensor.h>
#include <rascaline.h>

/// Compute SOAP power spectrum, this is the same code as the 'compute-soap'
/// example
static mts_tensormap_t* compute_soap(const char* path);

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

    mts_tensormap_t* descriptor = compute_soap(argv[1]);
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
    mts_tensormap_free(descriptor);

    if (got_error) {
        return 1;
    } else {
        return 0;
    }
}

static mts_tensormap_t* move_keys_to_samples(mts_tensormap_t* descriptor, const char* keys_to_move[], size_t keys_to_move_len);
static mts_tensormap_t* move_keys_to_properties(mts_tensormap_t* descriptor, const char* keys_to_move[], size_t keys_to_move_len);

// this is the same function as in the compute-soap.c example
mts_tensormap_t* compute_soap(const char* path) {
    int status = RASCAL_SUCCESS;
    rascal_calculator_t* calculator = NULL;
    rascal_system_t* systems = NULL;
    uintptr_t n_systems = 0;
    const double* values = NULL;
    const uintptr_t* shape = NULL;
    uintptr_t shape_count = 0;
    bool got_error = true;
    const char* keys_to_samples[] = {"center_type"};
    const char* keys_to_properties[] = {"neighbor_1_type", "neighbor_2_type"};

    // use the default set of options, computing all samples and all features
    rascal_calculation_options_t options = {0};
    const char* gradients_list[] = {"positions"};
    options.gradients = gradients_list;
    options.gradients_count = 1;

    mts_tensormap_t* descriptor = NULL;
    const mts_block_t* block = NULL;
    mts_array_t data = {0};
    mts_labels_t keys_to_move = {0};

    const char* parameters = "{\n"
        "\"cutoff\": {\n"
        "    \"radius\": 5.0,\n"
        "    \"smoothing\": {\"type\": \"ShiftedCosine\", \"width\": 0.5}\n"
        "},\n"
        "\"density\": {\n"
        "    \"type\": \"Gaussian\",\n"
        "    \"width\": 0.3\n"
        "},\n"
        "\"basis\": {\n"
        "    \"type\": \"TensorProduct\",\n"
        "    \"max_angular\": 6,\n"
        "    \"radial\": {\"type\": \"Gto\", \"max_radial\": 6}\n"
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

    descriptor = move_keys_to_samples(descriptor, keys_to_samples, 1);
    if (descriptor == NULL) {
        printf("Error: %s\n", mts_last_error());
        goto cleanup;
    }

    descriptor = move_keys_to_properties(descriptor, keys_to_properties, 2);
    if (descriptor == NULL) {
        printf("Error: %s\n", mts_last_error());
        goto cleanup;
    }

cleanup:
    rascal_calculator_free(calculator);
    rascal_basic_systems_free(systems, n_systems);

    return descriptor;
}


mts_tensormap_t* move_keys_to_samples(mts_tensormap_t* descriptor, const char* keys_to_move[], size_t keys_to_move_len) {
    mts_labels_t keys = {0};
    mts_tensormap_t* moved_descriptor = NULL;

    keys.names = keys_to_move;
    keys.size = keys_to_move_len;
    keys.values = NULL;
    keys.count = 0;

    moved_descriptor = mts_tensormap_keys_to_samples(descriptor, keys, true);
    mts_tensormap_free(descriptor);

    return moved_descriptor;
}


mts_tensormap_t* move_keys_to_properties(mts_tensormap_t* descriptor, const char* keys_to_move[], size_t keys_to_move_len) {
    mts_labels_t keys = {0};
    mts_tensormap_t* moved_descriptor = NULL;

    keys.names = keys_to_move;
    keys.size = keys_to_move_len;
    keys.values = NULL;
    keys.count = 0;

    moved_descriptor = mts_tensormap_keys_to_properties(descriptor, keys, true);
    mts_tensormap_free(descriptor);

    return moved_descriptor;
}
