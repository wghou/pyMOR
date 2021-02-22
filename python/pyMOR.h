#ifndef __PY_ELASTIC_FORCE_FEM_H
#define __PY_ELASTIC_FORCE_FEM_H

#include <elasticForceFEM.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class pyElasticForceFEM
{
    public:
    pyElasticForceFEM(py::array_t<float, py::array::c_style | py::array::forcecast> x, py::array_t<int, py::array::c_style | py::array::forcecast> tet)
    {
        fem = new ElasticForceFEM(x.data(), (int)(x.size()/3), tet.data(), (int)(tet.size()/4));
    }

    ~pyElasticForceFEM()
    {
        delete fem;
    }

    public:
    // setter
    void setDensity(float den)
    {
        fem->density = den;
    }

    void setMu01(float _mu01)
    {
        fem->mu01 = _mu01;
    }

    void setMu10(float _mu10)
    {
        fem->mu10 = _mu10;
    }

    void setV1(float _v1)
    {
        fem->v1 = _v1;
    }

    float getDensity()
    {
        return fem->density;
    }

    float getMu01()
    {
        return fem->mu01;
    }

    float getMu10()
    {
        return fem->mu10;
    }

    float gettV1()
    {
        return fem->v1;
    }

    py::array_t<float, py::array::c_style | py::array::forcecast> ComputeForces(py::array_t<float, py::array::c_style | py::array::forcecast> u, bool addGravity = false )
    {
        if( u.size() != fem->numVertices*3)
        {
            throw std::runtime_error("input size donnot match the mesh size.");
        }

        auto u_buffer = u.request();
        float *u_ptr = (float*)u_buffer.ptr;

        // allocate py::array (to pass the result of the C++ function to Python)
        auto result        = py::array_t<float, py::array::c_style | py::array::forcecast>(fem->numVertices*3);
        auto result_buffer = result.request();
        float *result_ptr    = (float *) result_buffer.ptr;

        fem->ComputeForces(u_ptr, result_ptr, addGravity);

        return result;
    }

    private:
    ElasticForceFEM* fem = nullptr;
};

#endif