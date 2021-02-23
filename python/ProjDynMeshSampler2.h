#ifndef PROJ_DYN_MESH_SAMPLER2_H
#define PROJ_DYN_MESH_SAMPLER2_H

#include <iostream>
#include "ProjUtils/ProjDynMeshSampler.h"
#include "ProjUtils/ProjDynutil.h"

class ProjDynMeshSampler2 : public ProjDynMeshSampler
{
    public:
    void init(PDPositions & pos, PDTriangles & tris, PDTets & tets)
    {
        ProjDynMeshSampler::init(pos, tris, tets);
    }

    PD::PDMatrix getRadialBaseFunctions(std::vector<unsigned int>& samples, double r, bool useIdicator = false)
    {
        double eps = -1.; 
        int numSmallSamples = -1; 
        double smallSamplesRadius = 1.;

        if (eps < 0) eps = std::sqrt(-std::log(BASE_FUNC_CUTOFF)) / r;

        PDScalar pi = 3.14159265358979323846;
        int nSamples = samples.size();
        PDMatrix baseFunctions;
        baseFunctions.setZero(m_nVertices, nSamples);
        PDScalar a = (1. / std::pow(r, 4.));
        PDScalar b = -2. * (1. / (r * r));

        for (int i = 0; i < nSamples; i++) {
            if (numSmallSamples > 0 && i > nSamples - numSmallSamples) {
                r = smallSamplesRadius;
                eps = std::sqrt(-std::log(BASE_FUNC_CUTOFF)) / r;
            }

            unsigned int curSample = samples[i];
            clearSources();
            addSource(curSample);
            computeDistances(r, false);

            for (int v = 0; v < m_nVertices; v++) {
                double curDist = getDistance(v);
                double val = 0;
                if (curDist < 0) {
                    val = 0;
                }
                else if (USE_QUARTIC_POL) {
                    if (curDist >= r) {
                        val = 0;
                    }
                    else {
                        val = a * std::pow(curDist, 4.) + b * (curDist * curDist) + 1;
                    }
                }
                else {
                    val = std::exp(-(curDist*eps*curDist*eps));
                    if (val < BASE_FUNC_CUTOFF) val = 0;
                }
                baseFunctions(v, i) = val;
            }

        }

        for (int v = 0; v < m_nVertices; v++) {
            PDScalar sum = baseFunctions.row(v).sum();
            if (sum < 1e-6) {
                std::cout << "Warning: a vertex isn't properly covered by any of the radial basis functions!" << std::endl;
                baseFunctions(v, indexOfMaxColCoeff(baseFunctions, v)) = 0.;
            }
            else {
                baseFunctions.row(v) /= sum;
            }
        }

        return baseFunctions;
    }
};

#endif