#include <rascaline.hpp>


int main() {
    try {
        auto calculator = rascaline::Calculator(
            "dummy_calculator",
            R"({"cutoff": 3.4, "delta": -3, "name": "testing", "gradients": true})"
        );
        return 0;
    } catch (const rascaline::RascalineError& e) {
        return 1;
    }
}
