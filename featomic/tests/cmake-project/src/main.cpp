#include <featomic.hpp>


int main() {
    try {
        auto calculator = featomic::Calculator(
            "dummy_calculator",
            R"({"cutoff": 3.4, "delta": -3, "name": "testing", "gradients": true})"
        );
        return 0;
    } catch (const featomic::FeatomicError& e) {
        return 1;
    }
}
