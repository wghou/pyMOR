#include "elasticForceFEM.h"
#include<string>

#define SVD_singularValue_eps 1e-8
#define INVERSIONTHRESHOLD          0.1 // max value

ElasticForceFEM::ElasticForceFEM(const double *x, int nodeNumber, const int* tet, int tetNumber)
{
	numVertices = nodeNumber;
	numElements = tetNumber;

	X = new double[nodeNumber * 3];
	Tet = new int[tetNumber * 4];

	memcpy(X, x, sizeof(double)*nodeNumber * 3);
	memcpy(Tet, tet, sizeof(int)*tetNumber * 4);


	dmInverses = (Mat3d*)malloc(sizeof(Mat3d) * numElements);
	for (int el = 0; el<numElements; el++)
	{
		int vaIndex = Tet[el * 4 + 0] * 3;
		int vbIndex = Tet[el * 4 + 1] * 3;
		int vcIndex = Tet[el * 4 + 2] * 3;
		int vdIndex = Tet[el * 4 + 3] * 3;

		Vec3d va(X[vaIndex], X[vaIndex + 1], X[vaIndex + 2]);
		Vec3d vb(X[vbIndex], X[vbIndex + 1], X[vbIndex + 2]);
		Vec3d vc(X[vcIndex], X[vcIndex + 1], X[vcIndex + 2]);
		Vec3d vd(X[vdIndex], X[vdIndex + 1], X[vdIndex + 2]);

		Vec3d dm1 = vd - va;
		Vec3d dm2 = vd - vb;
		Vec3d dm3 = vd - vc;

		Mat3d tmp(dm1[0], dm2[0], dm3[0], dm1[1], dm2[1], dm3[1], dm1[2], dm2[2], dm3[2]);
		//printf("--- dm ---\n");
		//tmp.print();
		dmInverses[el] = inv(tmp);
		//printf("--- inv(dm) ---\n");
		//dmInverses[e].print();
	}


	areaWeightedVertexNormals = (Vec3d*)malloc(sizeof(Vec3d) * numElements * 4);
	for (int el = 0; el < numElements; el++)
	{
		int vaIndex = Tet[el * 4 + 0] * 3;
		int vbIndex = Tet[el * 4 + 1] * 3;
		int vcIndex = Tet[el * 4 + 2] * 3;
		int vdIndex = Tet[el * 4 + 3] * 3;

		Vec3d va(X[vaIndex], X[vaIndex + 1], X[vaIndex + 2]);
		Vec3d vb(X[vbIndex], X[vbIndex + 1], X[vbIndex + 2]);
		Vec3d vc(X[vcIndex], X[vcIndex + 1], X[vcIndex + 2]);
		Vec3d vd(X[vdIndex], X[vdIndex + 1], X[vdIndex + 2]);

		// compute normals for the four faces: acb, adc, abd, bcd
		Vec3d acbNormal = cross(vc - va, vb - va);
		Vec3d adcNormal = cross(vd - va, vc - va);
		Vec3d abdNormal = cross(vb - va, vd - va);
		Vec3d bcdNormal = cross(vc - vb, vd - vb);

		// if the tet vertices abcd form a positive orientation, the normals are now correct
		// otherwise, we need to flip them
		double orientation = dot(vd - va, cross(vb - va, vc - va));
		if (orientation < 0)
		{
			acbNormal *= -1.0;
			adcNormal *= -1.0;
			abdNormal *= -1.0;
			bcdNormal *= -1.0;
		}

		// triangle area = 0.5 |u x v|
		double acbArea = 0.5 * sqrt(dot(acbNormal, acbNormal));
		double adcArea = 0.5 * sqrt(dot(adcNormal, adcNormal));
		double abdArea = 0.5 * sqrt(dot(abdNormal, abdNormal));
		double bcdArea = 0.5 * sqrt(dot(bcdNormal, bcdNormal));

		// normalize
		acbNormal.normalize();
		adcNormal.normalize();
		abdNormal.normalize();
		bcdNormal.normalize();

		areaWeightedVertexNormals[4 * el + 0] = (acbArea * acbNormal + adcArea * adcNormal + abdArea * abdNormal) / 3.0;
		areaWeightedVertexNormals[4 * el + 1] = (acbArea * acbNormal + abdArea * abdNormal + bcdArea * bcdNormal) / 3.0;
		areaWeightedVertexNormals[4 * el + 2] = (acbArea * acbNormal + adcArea * adcNormal + bcdArea * bcdNormal) / 3.0;
		areaWeightedVertexNormals[4 * el + 3] = (adcArea * adcNormal + abdArea * abdNormal + bcdArea * bcdNormal) / 3.0;

		/*
		printf("--- areaWeightedVertexNormals ---\n");
		printf("a = "); areaWeightedVertexNormals[4*el+0].print();
		printf("b = "); areaWeightedVertexNormals[4*el+1].print();
		printf("c = "); areaWeightedVertexNormals[4*el+2].print();
		printf("d = "); areaWeightedVertexNormals[4*el+3].print();
		*/
	}
}

ElasticForceFEM::ElasticForceFEM(const float *x, int nodeNumber, const int* tet, int tetNumber)
{
	numVertices = nodeNumber;
	numElements = tetNumber;

	X = new double[nodeNumber * 3];
	Tet = new int[tetNumber * 4];

	for (int i = 0; i < nodeNumber; i++)
	{
		X[i * 3 + 0] = x[i * 3 + 0];
		X[i * 3 + 1] = x[i * 3 + 1];
		X[i * 3 + 2] = x[i * 3 + 2];
	}
	//memcpy(X, x, sizeof(double)*nodeNumber * 3);
	
	memcpy(Tet, tet, sizeof(int)*tetNumber * 4);


	dmInverses = (Mat3d*)malloc(sizeof(Mat3d) * numElements);
	for (int el = 0; el<numElements; el++)
	{
		int vaIndex = Tet[el * 4 + 0] * 3;
		int vbIndex = Tet[el * 4 + 1] * 3;
		int vcIndex = Tet[el * 4 + 2] * 3;
		int vdIndex = Tet[el * 4 + 3] * 3;

		Vec3d va(X[vaIndex], X[vaIndex + 1], X[vaIndex + 2]);
		Vec3d vb(X[vbIndex], X[vbIndex + 1], X[vbIndex + 2]);
		Vec3d vc(X[vcIndex], X[vcIndex + 1], X[vcIndex + 2]);
		Vec3d vd(X[vdIndex], X[vdIndex + 1], X[vdIndex + 2]);

		Vec3d dm1 = vd - va;
		Vec3d dm2 = vd - vb;
		Vec3d dm3 = vd - vc;

		Mat3d tmp(dm1[0], dm2[0], dm3[0], dm1[1], dm2[1], dm3[1], dm1[2], dm2[2], dm3[2]);
		//printf("--- dm ---\n");
		//tmp.print();
		dmInverses[el] = inv(tmp);
		//printf("--- inv(dm) ---\n");
		//dmInverses[e].print();
	}


	areaWeightedVertexNormals = (Vec3d*)malloc(sizeof(Vec3d) * numElements * 4);
	for (int el = 0; el < numElements; el++)
	{
		int vaIndex = Tet[el * 4 + 0] * 3;
		int vbIndex = Tet[el * 4 + 1] * 3;
		int vcIndex = Tet[el * 4 + 2] * 3;
		int vdIndex = Tet[el * 4 + 3] * 3;

		Vec3d va(X[vaIndex], X[vaIndex + 1], X[vaIndex + 2]);
		Vec3d vb(X[vbIndex], X[vbIndex + 1], X[vbIndex + 2]);
		Vec3d vc(X[vcIndex], X[vcIndex + 1], X[vcIndex + 2]);
		Vec3d vd(X[vdIndex], X[vdIndex + 1], X[vdIndex + 2]);

		// compute normals for the four faces: acb, adc, abd, bcd
		Vec3d acbNormal = cross(vc - va, vb - va);
		Vec3d adcNormal = cross(vd - va, vc - va);
		Vec3d abdNormal = cross(vb - va, vd - va);
		Vec3d bcdNormal = cross(vc - vb, vd - vb);

		// if the tet vertices abcd form a positive orientation, the normals are now correct
		// otherwise, we need to flip them
		double orientation = dot(vd - va, cross(vb - va, vc - va));
		if (orientation < 0)
		{
			acbNormal *= -1.0;
			adcNormal *= -1.0;
			abdNormal *= -1.0;
			bcdNormal *= -1.0;
		}

		// triangle area = 0.5 |u x v|
		double acbArea = 0.5 * sqrt(dot(acbNormal, acbNormal));
		double adcArea = 0.5 * sqrt(dot(adcNormal, adcNormal));
		double abdArea = 0.5 * sqrt(dot(abdNormal, abdNormal));
		double bcdArea = 0.5 * sqrt(dot(bcdNormal, bcdNormal));

		// normalize
		acbNormal.normalize();
		adcNormal.normalize();
		abdNormal.normalize();
		bcdNormal.normalize();

		areaWeightedVertexNormals[4 * el + 0] = (acbArea * acbNormal + adcArea * adcNormal + abdArea * abdNormal) / 3.0;
		areaWeightedVertexNormals[4 * el + 1] = (acbArea * acbNormal + abdArea * abdNormal + bcdArea * bcdNormal) / 3.0;
		areaWeightedVertexNormals[4 * el + 2] = (acbArea * acbNormal + adcArea * adcNormal + bcdArea * bcdNormal) / 3.0;
		areaWeightedVertexNormals[4 * el + 3] = (adcArea * adcNormal + abdArea * abdNormal + bcdArea * bcdNormal) / 3.0;

		/*
		printf("--- areaWeightedVertexNormals ---\n");
		printf("a = "); areaWeightedVertexNormals[4*el+0].print();
		printf("b = "); areaWeightedVertexNormals[4*el+1].print();
		printf("c = "); areaWeightedVertexNormals[4*el+2].print();
		printf("d = "); areaWeightedVertexNormals[4*el+3].print();
		*/
	}
}

ElasticForceFEM::~ElasticForceFEM()
{
	delete[] X;
	delete[] Tet;

	free(dmInverses);
	free(areaWeightedVertexNormals);
}

void ElasticForceFEM::SetMaterialParams(double den, double _mu01, double _mu10, double _v1)
{
	density = den; mu01 = _mu01; mu10 = _mu10; v1 = _v1;
}

void AddCompressionResistanceGradient(int elementIndex, double * invariants, double * gradient, bool enableCompressionResistance = false)
{
	if (enableCompressionResistance)
	{
		double IIIC = invariants[2];
		double J = sqrt(IIIC);

		if (J < 1)
		{
			gradient[2] += -(J - 1.0) * (J - 1.0) / (1728.0 * J);
		}
	}
}


// HomogeneousMooneyRivlinIsotropicMaterial::
void ElasticForceFEM::ComputeEnergyGradient(int elementIndex, double * invariants, double * gradient) // invariants and gradient are 3-vectors
{
	double Ic = invariants[0];
	double IIc = invariants[1];
	double IIIc = invariants[2];
	gradient[0] = (Ic * mu01) / pow(IIIc, 2.0 / 3.0) +
		mu10 / pow(IIIc, 1.0 / 3.0);
	gradient[1] = (-0.5 * mu01) / pow(IIIc, 2.0 / 3.0);
	gradient[2] = (-1.0 / 3.0 * (Ic * Ic - IIc) * mu01) / pow(IIIc, 5.0 / 3.0) -
		(1.0 / 3.0 * Ic * mu10) / pow(IIIc, 4.0 / 3.0) +
		((-1.0 + sqrt(IIIc)) * v1) / sqrt(IIIc);

	//AddCompressionResistanceGradient(elementIndex, invariants, gradient);
}

void ElasticForceFEM::ComputeDiagonalPFromStretches(int elementIndex, double * lambda, double * PDiag)
{
	double invariants[3];

	double lambda2[3] = { lambda[0] * lambda[0], lambda[1] * lambda[1], lambda[2] * lambda[2] };
	double IC = lambda2[0] + lambda2[1] + lambda2[2];
	double IIC = lambda2[0] * lambda2[0] + lambda2[1] * lambda2[1] + lambda2[2] * lambda2[2];
	double IIIC = lambda2[0] * lambda2[1] * lambda2[2];

	invariants[0] = IC;
	invariants[1] = IIC;
	invariants[2] = IIIC;

	double dPsidI[3];

	ComputeEnergyGradient(elementIndex, invariants, dPsidI);

	// PDiag = [ dI / dlambda ]^T * dPsidI

	double mat[9];
	mat[0] = 2.0 * lambda[0];
	mat[1] = 2.0 * lambda[1];
	mat[2] = 2.0 * lambda[2];
	mat[3] = 4.0 * lambda[0] * lambda[0] * lambda[0];
	mat[4] = 4.0 * lambda[1] * lambda[1] * lambda[1];
	mat[5] = 4.0 * lambda[2] * lambda[2] * lambda[2];
	mat[6] = 2.0 * lambda[0] * lambda2[1] * lambda2[2];
	mat[7] = 2.0 * lambda[1] * lambda2[0] * lambda2[2];
	mat[8] = 2.0 * lambda[2] * lambda2[0] * lambda2[1];

	Mat3d matM(mat);
	Vec3d dPsidIV(dPsidI);
	Vec3d result;

	result = trans(matM) * dPsidIV;
	result.convertToArray(PDiag);
}

void ElasticForceFEM::ComputeForces(float * u, float * internalForces, bool addGravity)
{
	if (addGravity)
	{
		for (int i = 0; i<numVertices; i++)
		{
			internalForces[3 * i + 0] = 0.0;
			internalForces[3 * i + 1] = -9.81; // gravity acts in negative-y direction; internal forces are opposite of external forces
			internalForces[3 * i + 2] = 0.0;
		}
	}
	else
	{
		// zero out the forces
		memset(internalForces, 0, sizeof( float) * numVertices * 3);
	}


	for (int el = 0; el < numElements; el++)
	{
		/*
		Compute the deformation gradient F.
		F = Ds * inv(Dm), where Ds is a 3x3 matrix where
		the columns are edge vectors of a tet in the current deformation,
		and Dm is a 3x3 matrix where the columns are edge vectors of a tet in
		the rest configuration. See p3 section 3 of [Irving 04] for more details.
		*/
		int vaIndex = Tet[el * 4 + 0] * 3;
		int vbIndex = Tet[el * 4 + 1] * 3;
		int vcIndex = Tet[el * 4 + 2] * 3;
		int vdIndex = Tet[el * 4 + 3] * 3;

		Vec3d va(u[vaIndex], u[vaIndex + 1], u[vaIndex + 2]);
		Vec3d vb(u[vbIndex], u[vbIndex + 1], u[vbIndex + 2]);
		Vec3d vc(u[vcIndex], u[vcIndex + 1], u[vcIndex + 2]);
		Vec3d vd(u[vdIndex], u[vdIndex + 1], u[vdIndex + 2]);

		Vec3d ds1 = vd - va;
		Vec3d ds2 = vd - vb;
		Vec3d ds3 = vd - vc;

		Mat3d tmp(ds1[0], ds2[0], ds3[0], ds1[1], ds2[1], ds3[1], ds1[2], ds2[2], ds3[2]);
		Mat3d F = tmp * dmInverses[el];
		/*
		The deformation gradient has now been computed and is available in Fs[el]
		*/

		// perform modified SVD on the deformation gradient
		Mat3d U, V;
		Vec3d Fhat;
		int modifiedSVD = 1;
		if (SVD(F, U, Fhat, V, SVD_singularValue_eps, modifiedSVD) != 0)
		{
			printf("error in diagonalization, el=%d\n", el);
			return;
		}

		/*
		SVD for the deformation gradient has now been computed.
		It is available in Us[el], Fhats[el], Vs[el].
		*/

		// clamp fHat if below the principal stretch threshold
		double fHat[3];
		double inversionThreshold = INVERSIONTHRESHOLD;
		for (int i = 0; i < 3; i++)
		{
			if (Fhat[i] < inversionThreshold)
			{
				//dropBelowThreshold = true;
				Fhat[i] = inversionThreshold;
			}
		}
		fHat[0] = Fhat[0];
		fHat[1] = Fhat[1];
		fHat[2] = Fhat[2];


		/*
		--- Now compute the internal forces ---

		The first Piola-Kirchhoff stress P is calculated by equation 1
		in p3 section 5 of [Irving 04]. Once we have P, we can compute
		the nodal forces G=PBm as described in section 4 of [Irving 04]
		*/

		double pHat[3];
		ComputeDiagonalPFromStretches(el, fHat, pHat); // calls the isotropic material to compute the diagonal P tensor, given the principal stretches in fHat
		Vec3d pHatv(pHat);

		// This is the 1st equation in p3 section 5 of [Irving 04]
		// P = Us[el] * diag(pHat) * trans(Vs[el])
		Mat3d P = U;
		P.multiplyDiagRight(pHatv);
		P = P * trans(V);

		//printf("--- P ---\n");
		//P.print();

		/*
		we compute the nodal forces by G=PBm as described in
		section 4 of [Irving 04]
		*/
		// multiply by 4 because each tet has 4 vertices
		Vec3d forceUpdateA = P * areaWeightedVertexNormals[4 * el + 0];
		Vec3d forceUpdateB = P * areaWeightedVertexNormals[4 * el + 1];
		Vec3d forceUpdateC = P * areaWeightedVertexNormals[4 * el + 2];
		Vec3d forceUpdateD = P * areaWeightedVertexNormals[4 * el + 3];

		internalForces[vaIndex + 0] += forceUpdateA[0];
		internalForces[vaIndex + 1] += forceUpdateA[1];
		internalForces[vaIndex + 2] += forceUpdateA[2];

		internalForces[vbIndex + 0] += forceUpdateB[0];
		internalForces[vbIndex + 1] += forceUpdateB[1];
		internalForces[vbIndex + 2] += forceUpdateB[2];

		internalForces[vcIndex + 0] += forceUpdateC[0];
		internalForces[vcIndex + 1] += forceUpdateC[1];
		internalForces[vcIndex + 2] += forceUpdateC[2];

		internalForces[vdIndex + 0] += forceUpdateD[0];
		internalForces[vdIndex + 1] += forceUpdateD[1];
		internalForces[vdIndex + 2] += forceUpdateD[2];
	}


}

