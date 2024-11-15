#include <stdio.h>
#include <featomic.h>


int main(void) {
    featomic_calculator_t* calculator = featomic_calculator(
        "dummy_calculator",
        "{\"cutoff\": 3.4, \"delta\": -3, \"name\": \"testing\", \"gradients\": true}"
    );

    if (calculator == NULL) {
        printf("error: %s\n", featomic_last_error());
        return 1;
    }

    featomic_calculator_free(calculator);

    return 0;
}
