#include <torch/script.h>

#include "rascaline/torch.hpp"
using namespace rascaline_torch;

TORCH_LIBRARY(rascaline, module) {
    // There is no way to access the docstrings from Python, so we don't bother
    // setting them to something useful here.
    const std::string DOCSTRING;

    module.class_<SystemHolder>("System")
        .def(torch::init<torch::Tensor, torch::Tensor, torch::Tensor>(),
            DOCSTRING,
            {torch::arg("species"), torch::arg("positions"), torch::arg("cell")}
        )
        .def("__str__", &SystemHolder::str)
        .def("__repr__", &SystemHolder::str)
        .def("__len__", &SystemHolder::len)
        .def_property("species", &SystemHolder::get_species)
        .def_property("positions", &SystemHolder::get_positions)
        .def_property("cell", &SystemHolder::get_cell)
        ;

    module.class_<CalculatorOptionsHolder>("CalculatorOptions")
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

    module.class_<CalculatorHolder>("CalculatorHolder")
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
            [](const TorchCalculator& self) -> std::tuple<std::string, std::string> {
                return {self->c_name(), self->parameters()};
            },
            // __setstate__
            [](std::tuple<std::string, std::string> state) -> TorchCalculator {
                return c10::make_intrusive<CalculatorHolder>(
                    std::get<0>(state), std::get<1>(state)
                );
            })
        ;

    module.def(
        "register_autograd("
            "__torch__.torch.classes.rascaline.System[] systems,"
            "__torch__.torch.classes.metatensor.TensorMap precomputed,"
            "str[] forward_gradients"
        ") -> __torch__.torch.classes.metatensor.TensorMap",
        register_autograd
    );
}
