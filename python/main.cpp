#include <pybind11/pybind11.h>
#include <pyElasticForceFEM.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(pyvegafem, m) {
    m.doc() = R"pbdoc(
        Pybind11 pyvegafem plugin
        -----------------------

        .. currentmodule:: pyvegafem

        .. autosummary::
           :toctree: _generate
    )pbdoc";

    py::class_<pyElasticForceFEM>(m, "FEM")
        .def(py::init<py::array_t<float, py::array::c_style | py::array::forcecast>, py::array_t<int, py::array::c_style | py::array::forcecast>>())
        .def("setDensity", &pyElasticForceFEM::setDensity)
        .def("setMu01", &pyElasticForceFEM::setMu01)
        .def("setMu10", &pyElasticForceFEM::setMu10)
        .def("setV1", &pyElasticForceFEM::setV1)
        .def("getDensity", &pyElasticForceFEM::getDensity)
        .def("getMu01", &pyElasticForceFEM::getMu01)
        .def("getMu10", &pyElasticForceFEM::getMu10)
        .def("gettV1", &pyElasticForceFEM::gettV1)
        .def("ComputeForces", &pyElasticForceFEM::ComputeForces);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
