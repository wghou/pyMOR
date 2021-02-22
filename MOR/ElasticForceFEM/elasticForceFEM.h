#pragma once
#ifndef ELASTICFORCEFEM_H
#define ELASTICFORCEFEM_H

#include"vec3d.h"
#include"mat3d.h"

class pyElasticForceFEM;

class ElasticForceFEM
{
	friend pyElasticForceFEM;
public:
	ElasticForceFEM(const double *x, int nodeNumber, const int* tet, int tetNumber);
	ElasticForceFEM(const float *x, int nodeNumber, const int* tet, int tetNumber);
	~ElasticForceFEM();
	void ComputeForces(float * u, float * internalForces, bool addGravity = false);
	void SetMaterialParams(double den, double _mu01, double _mu10, double _v1);
private:
	void ComputeEnergyGradient(int elementIndex, double * invariants, double * gradient);
	void ComputeDiagonalPFromStretches(int elementIndex, double * lambda, double * PDiag);

private:
	int numVertices = 0;
	int numElements = 0;

	int* Tet;
	double *X;

	Mat3d * dmInverses;
	Vec3d * areaWeightedVertexNormals;

	double density = 1000;
	double mu01 = 1e6;
	double mu10 = 1e6;
	double v1 = 1e6;
};

#endif // !ELASTICFORCEFEM_H
