#include "Fluid.cuh"

#include <thrust/transform.h>
#include <device_launch_parameters.h>

#pragma once

void employGravityCuImpl(double dt, double gravity, v1d& v);

__global__ void advectVelCuImpl(
	double dt, int nx, int ny, int nz, double dxyz, int threadCnt,
	double* barrier, double* u, double* v, double* w,
	double* newU, double* newV, double* newW);

__global__ void advectSmokeCuImpl(
	double dt, int nx, int ny, int nz, double dxyz, int threadCnt,
	double* barrier, double* u, double* v,double* w, double* smoke, double* newSmoke);

__global__ void clearToZeroCuImpl(int size, int threadCnt, double* arr);

__global__ void setObstacleCuImpl(
	double vx, double vy, int nx, int ny, int nz,
	double dxy, int threadCnt, double ballX, double ballY,
	double ballR2, double t, double* barrier, double* u, double* v, double* w, double* smoke);

__global__ void initSceneVortexStreetCuImpl(
	int nx, int ny, int nz, double dx, double dy, double dz, int threadCnt,
	double inVel, double pipeMinH, double pipeMaxH,
	double* barrier, double* smoke, double* u);

void boundryCondPeriodCuImpl(
	int nx, int ny, int nz, double* u, double* v, double* w);

void ConjugateGradientCu(
	int maxIter, double dt, int nx, int ny, int nz, double pdxydt,
	v1d& barrier, v1d& u, v1d& v, v1d& w, v1d& pressure);

__host__ void GaussSedielRBCu(
	int iter, double dt, int nx, int ny, int nz, double pdxydt,
	double* barrier, double* u, double* v, double* w, double* pre);

#define rawPtr1(a)         thrust::raw_pointer_cast(##a.data())
#define rawPtr2(a,b)       rawPtr1(a),rawPtr1(b)
#define rawPtr3(a,b,c)     rawPtr1(a),rawPtr1(b),rawPtr1(c)
#define rawPtr4(a,b,c,d)   rawPtr1(a),rawPtr1(b),rawPtr1(c),rawPtr1(d)
#define rawPtr5(a,b,c,d,e) rawPtr1(a),rawPtr1(b),rawPtr1(c),rawPtr1(d),rawPtr1(e)
