#include <torch/script.h>

#include "rascaline/torch/system.hpp"
#include "rascaline/torch/calculator.hpp"
using namespace rascaline_torch;

TORCH_LIBRARY(rascaline, m) {
    // There is no way to access the docstrings from Python, so we don't bother
    // setting them to something useful here.
    const std::string DOCSTRING = "";

    m.class_<SystemHolder>("System")
        .def(torch::init<torch::Tensor, torch::Tensor, torch::Tensor>(),
            DOCSTRING,
            {torch::arg("species"), torch::arg("positions"), torch::arg("cell")}
        )
        .def("__str__", &SystemHolder::__str__)
        .def("__repr__", &SystemHolder::__str__)
        .def("__len__", &SystemHolder::__len__)
        .def_property("species", &SystemHolder::get_species)
        .def_property("positions", &SystemHolder::get_positions)
        .def_property("cell", &SystemHolder::get_cell)
        ;

    m.class_<CalculatorOptionsHolder>("CalculatorOptions")
        .def(torch::init())
        .def_readwrite("gradients", &CalculatorOptionsHolder::gradients)
        .def_property("selected_keys",
            &CalculatorOptionsHolder::selected_keys,
            &CalculatorOptionsHolder::set_selected_keys
        )
        .def_property("selected_samples",
            &CalculatorOptionsHolder::selected_samples,
            &CalculatorOptionsHolder::set_selected_samples
        )
        .def_property("selected_properties",
            &CalculatorOptionsHolder::selected_properties,
            &CalculatorOptionsHolder::set_selected_properties
        )
        ;

    m.class_<CalculatorHolder>("CalculatorHolder")
        .def(torch::init<std::string, std::string>(),
            DOCSTRING,
            {torch::arg("name"), torch::arg("parameters")}
        )
        .def_property("name", &CalculatorHolder::name)
        .def_property("parameters", &CalculatorHolder::parameters)
        .def_property("cutoffs", &CalculatorHolder::cutoffs)
        .def("compute", &CalculatorHolder::compute, DOCSTRING, {
            torch::arg("systems"),
            torch::arg("options") = {}
        })
        .def_pickle(
            // __getstate__
            [](const TorchCalculator& self) -> std::vector<std::string> {
                return {self->name(), self->parameters()};
            },
            // __setstate__
            [](std::vector<std::string> state) -> TorchCalculator {
                return c10::make_intrusive<CalculatorHolder>(
                    state[0], state[1]
                );
            })
        ;

    m.def(
        "register_autograd("
            "__torch__.torch.classes.rascaline.System[] systems,"
            "__torch__.torch.classes.metatensor.TensorMap precomputed,"
            "str[] forward_gradients"
        ") -> __torch__.torch.classes.metatensor.TensorMap",
        register_autograd
    );
}
