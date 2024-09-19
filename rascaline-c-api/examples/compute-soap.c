#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include <rascaline.h>
#include <metatensor.h>

static mts_tensormap_t* move_keys_to_samples(mts_tensormap_t* descriptor, const char* keys_to_move[], size_t keys_to_move_len);
static mts_tensormap_t* move_keys_to_properties(mts_tensormap_t* descriptor, const char* keys_to_move[], size_t keys_to_move_len);

int main(int argc, char* argv[]) {
    int status = RASCAL_SUCCESS;
    rascal_calculator_t* calculator = NULL;
    rascal_system_t* systems = NULL;
    uintptr_t n_systems = 0;
    double* values = NULL;
    const uintptr_t* shape = NULL;
    uintptr_t shape_count = 0;
    bool got_error = true;
    const char* keys_to_samples[] = {"center_type"};
    const char* keys_to_properties[] = {"neighbor_1_type", "neighbor_2_type"};
    // use the default set of options, computing all samples and all features,
    // and including gradients with respect to positions
    rascal_calculation_options_t options = {0};
    const char* gradients_list[] = {"positions"};
    options.gradients = gradients_list;
    options.gradients_count = 1;

    mts_tensormap_t* descriptor = NULL;
    mts_block_t* block = NULL;
    mts_array_t array = {0};

    // hyper-parameters for the calculation as JSON
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

    // The descriptor is a metatensor `TensorMap`, containing multiple blocks.
    // We can transform it to a single block containing a dense representation,
    // with one sample for each atom-centered environment.
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

    // extract the unique block and corresponding values from the descriptor
    status = mts_tensormap_block_by_id(descriptor, &block, 0);
    if (status != MTS_SUCCESS) {
        printf("Error: %s\n", mts_last_error());
        goto cleanup;
    }

    status = mts_block_data(block, &array);
    if (status != MTS_SUCCESS) {
        printf("Error: %s\n", mts_last_error());
        goto cleanup;
    }

    // callback the functions on the mts_array_t to extract the shape/data pointer
    status = array.shape(array.ptr, &shape, &shape_count);
    if (status != MTS_SUCCESS) {
        printf("Error: %s\n", mts_last_error());
        goto cleanup;
    }

    status = array.data(array.ptr, &values);
    if (status != MTS_SUCCESS) {
        printf("Error: %s\n", mts_last_error());
        goto cleanup;
    }

    if (status != MTS_SUCCESS) {
        printf("Error: %s\n", mts_last_error());
        goto cleanup;
    }
    assert(shape_count == 2);

    // you can now use `values` as the input of a machine learning algorithm
    printf("the value array shape is %lu x %lu\n", shape[0], shape[1]);

    got_error = false;
cleanup:
    mts_tensormap_free(descriptor);
    rascal_calculator_free(calculator);
    rascal_basic_systems_free(systems, n_systems);

    if (got_error) {
        return 1;
    } else {
        return 0;
    }
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
