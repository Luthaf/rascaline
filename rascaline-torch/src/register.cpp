#include <torch/script.h>

#include "rascaline/torch/system.hpp"
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
        .def("__len__", &SystemHolder::__len__)
        .def_property("species", &SystemHolder::get_species)
        .def_property("positions", &SystemHolder::get_positions)
        .def_property("cell", &SystemHolder::get_cell)
        ;
}
