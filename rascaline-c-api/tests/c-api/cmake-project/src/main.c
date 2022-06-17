#include <stdio.h>

#include <rascaline.h>


int main(void) {
    rascal_calculator_t* calculator = rascal_calculator(
        "dummy_calculator",
        "{\"cutoff\": 3.4, \"delta\": -3, \"name\": \"testing\", \"gradients\": true}"
    );

    if (calculator == NULL) {
        printf("error: %s\n", rascal_last_error());
        return 1;
    }

    rascal_calculator_free(calculator);

    return 0;
}
