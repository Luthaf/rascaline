#include "featomic.hpp"
#include "catch.hpp"


class BadSystem: public featomic::System {
public:
    uintptr_t size() const override {
        throw std::runtime_error("unimplemented function 'size'");
    }

    const int32_t* types() const override {
        throw std::runtime_error("unimplemented function 'types'");
    }

    const double* positions() const override {
        throw std::runtime_error("unimplemented function 'positions'");
    }

    CellMatrix cell() const override {
        throw std::runtime_error("unimplemented function 'cell'");
    }

    void compute_neighbors(double cutoff) override {
        throw std::runtime_error("unimplemented function 'compute_neighbors'");
    }

    const std::vector<featomic_pair_t>& pairs() const override {
        throw std::runtime_error("unimplemented function 'pairs'");
    }

    const std::vector<featomic_pair_t>& pairs_containing(uintptr_t atom) const override {
        throw std::runtime_error("unimplemented function 'pairs_containing'");
    }
};

TEST_CASE("systems errors") {
    const char* HYPERS_JSON = R"({
        "cutoff": 3.0,
        "delta": 4,
        "name": "",
        "gradients": true
    })";

    auto system = BadSystem();
    auto calculator = featomic::Calculator("dummy_calculator", HYPERS_JSON);

    CHECK_THROWS_WITH(calculator.compute(system), "unimplemented function 'types'");
}
