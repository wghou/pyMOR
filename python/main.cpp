#include <pybind11/pybind11.h>
#include <pyMOR.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(pymor, m) {
    m.doc() = R"pbdoc(
        Pybind11 pymor plugin
        -----------------------

        .. currentmodule:: pymor

        .. autosummary::
           :toctree: _generate
    )pbdoc";

    py::class_<pyMOR>(m, "MOR")
        .def(py::init<PDPositions, PDTriangles, PDTets>())
        .def(py::init<PDPositions, PDVectori>())
        .def("createSkinningSpace", &pyMOR::createSkinningSpace)
        .def("projectToSubspace", &pyMOR::projectToSubspace)
        .def("projectFromSubspace", &pyMOR::projectFromSubspace);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
