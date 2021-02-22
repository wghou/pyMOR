#ifndef __PY_ELASTIC_FORCE_FEM_H
#define __PY_ELASTIC_FORCE_FEM_H

#include "ProjUtils/ProjDynMeshSampler.h"
#include "ProjUtils/ProjDynUtil.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;
using namespace PD;

#define PROJ_DYN_MIN_MASS 1e-10f

class pyMOR
{
    public:
    pyMOR(PDPositions x, PDTriangles facet, PDTets tet)
    {
        m_positions = x;
        m_triangles = facet;
        m_tetrahedrons = tet;

        m_sampler.init(m_positions, m_triangles, m_tetrahedrons);

        PDVector m_vertexMasses = vertexMasses(m_tetrahedrons, m_positions);
        std::vector<Eigen::Triplet<PDScalar>> massEntries;
        massEntries.reserve(m_positions.rows());
        for (int v = 0; v < m_positions.rows(); v++) {
            massEntries.push_back(Eigen::Triplet<PDScalar>(v, v, m_vertexMasses(v)));
        }
        m_massMatrix = PDSparseMatrix(m_positions.rows(), m_positions.rows());
        m_massMatrix.setFromTriplets(massEntries.begin(), massEntries.end());
    }

    virtual ~pyMOR() {}

    public:
    // step 2: create skinning space
    std::vector<unsigned int> createSkinningSpace(int numSamples)
    {
        // 选取采样
        m_samples = m_sampler.getSamples(numSamples);

        numSamples = m_samples.size();
	    std::sort(m_samples.begin(), m_samples.end());
	    m_samples.erase(std::unique(m_samples.begin(), m_samples.end()), m_samples.end());

        PDScalar furthestDist = m_sampler.getSampleDiameter(m_samples);
	    PDScalar r = furthestDist * m_baseFunctionRadius;

        m_baseFunctionWeights = m_sampler.getRadialBaseFunctions(m_samples, r, false);

        bool isFlat = false;
		if (m_positions.col(2).norm() < 1e-10) isFlat = true;
        m_baseFunctions = internalCreateSkinningSpace(m_positions, m_baseFunctionWeights, isFlat);

        m_baseFunctionsTransposed = m_baseFunctions.transpose();

        return m_samples;
    }

    // step 3: project pos to subspace
    void projectToSubspace(PDPositions & pos)
    {
        PDMatrix L = m_baseFunctionsTransposed * m_massMatrix * m_baseFunctions;
		m_subspaceSolver.compute(L);
        PDPositions rhs = m_baseFunctionsTransposed * m_massMatrix * pos;
        m_positionsSubspace.resize(m_baseFunctionsTransposed.rows(), 3);
        for (int d = 0; d < 3; d++) {
            m_positionsSubspace.col(d) = m_subspaceSolver.solve(rhs.col(d));
        }
    }

    // step 4: project subspace to pos
    PDPositions projectFromSubspace()
    {
        return m_baseFunctions * m_positionsSubspace;
    }

    protected:
    PDMatrix internalCreateSkinningSpace(PDPositions& restPositions, PDMatrix& weights, bool flatSpace)
	{
		if (weights.hasNaN() || restPositions.hasNaN()) {
			std::cout << "Warning: weights or rest-state used to create skinning space have NaN values!" << std::endl;
            throw std::runtime_error("Warning: weights or rest-state used to create skinning space have NaN values!");
        }

		int numGroups = weights.cols();
		int numRows = restPositions.rows();
		int dim = restPositions.cols() - (flatSpace ? 1 : 0);
		PDMatrix skinningSpace(numRows, numGroups * (dim + 1) + 1);

		bool error = false;
        for (int v = 0; v < numRows; v++) {
            for (int g = 0; g < numGroups; g++) {
                double curWeight = 0;
                curWeight = weights(v, g);
                if (std::isnan(curWeight)) {
                    std::cout << "Warning: NaN weight during skinning space construction!" << std::endl;
                    throw std::runtime_error("Warning: NaN weight during skinning space construction!");
                    error = true;
                    break;
                }
                for (int d = 0; d < dim; d++) {
                    skinningSpace(v, g * (dim + 1) + d) = restPositions(v, d) * curWeight;
                }
                skinningSpace(v, g*(dim + 1) + dim) = curWeight;
            }
        }

		if (error) {
			return PDMatrix(0, 0);
		}

		skinningSpace.col(skinningSpace.cols() - 1).setConstant(1.);

		if (skinningSpace.hasNaN()) {
			std::cout << "Warning: NaN entry in constructed skinning space!" << std::endl;
            throw std::runtime_error("Warning: NaN entry in constructed skinning space!");
		}

		return skinningSpace;
	}

    PDVector vertexMasses(PDTets& tets, PDPositions& positions) {
		PDVector vMasses(positions.rows());
		vMasses.fill(0);
		int numTets = tets.rows();
		for (int tInd = 0; tInd < numTets; tInd++) {
			PDScalar curArea = tetArea(tets.row(tInd), positions) * (1. / 4.);

			vMasses(tets(tInd, 0), 0) += curArea;
			vMasses(tets(tInd, 1), 0) += curArea;
			vMasses(tets(tInd, 2), 0) += curArea;
			vMasses(tets(tInd, 3), 0) += curArea;
		}

		for (int vInd = 0; vInd < vMasses.rows(); vInd++) {
			if (vMasses(vInd, 0) < PROJ_DYN_MIN_MASS) {
				vMasses(vInd, 0) = PROJ_DYN_MIN_MASS;
			}
		}

		return vMasses;
	}

    private:
        // 模型网格初始位置
    PDPositions m_positions;
    PDTriangles m_triangles;
    PDTets m_tetrahedrons;

    // 子空间
    ProjDynMeshSampler m_sampler;
    std::vector< unsigned int > m_samples;
    PDPositions m_positionsSubspace;

    // 参数
    PDScalar m_baseFunctionRadius = 1.1; // The larger this number, the larger the support of the base functions.
    PDSparseMatrix m_massMatrix;
    PDMatrix m_baseFunctionWeights;

    PDMatrix m_baseFunctions;
	PDMatrix m_baseFunctionsTransposed;
 
    Eigen::LLT<PDMatrix> m_subspaceSolver;  
};

#endif