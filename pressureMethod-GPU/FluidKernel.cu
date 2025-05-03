#include "FluidKernel.cuh"

#define myMax(a, b) ((a) > (b) ? (a) : (b))
#define myMin(a, b) ((a) < (b) ? (a) : (b))

void employGravityCuImpl(double dt, double gravity, v1d& v)
{
	if (fabs(gravity) < 1e-6)return;
	thrust::transform(
		v.begin(), v.end(), v.begin(),[=]__device__(int i) { return v[i] + gravity * dt; }
	);
}

__global__ void setObstacleCuImpl(
	double vx, double vy, int nx, int ny, int nz,
	double dxy, int threadCnt, double ballX, double ballY,
	double ballR2, double t, double* barrier, double* u, double* v, double* w, double* smoke)
{
	const int sz = (nx - 2) * (ny - 2) * (nz - 2);
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sz; idx += threadCnt)
	{
		int x = idx % (nx - 2) + 1;
		int y = (idx / (nx - 2)) % (ny - 2) + 1;
		int z = idx / ((nx - 2) * (ny - 2)) + 1;

		if (y >= ny - 2 || x >= nx - 2 || z >= nz - 2)continue;

		unsigned baseIdx = z * nx * ny + y * nx + x;
		barrier[baseIdx] = 1.0;

		double dx1 = (x + 0.5) * dxy - ballX;
		double dy1 = (y + 0.5) * dxy - ballY;

		if (dx1 * dx1 + dy1 * dy1 < ballR2)
		{
			barrier[baseIdx] = 0;
			smoke[baseIdx] = (0.5 + 0.5 * sin(t));
			u[baseIdx + 1] = u[baseIdx] = vx;
			v[baseIdx + nx] = v[baseIdx] = vy;
		}
	}
}

__device__ double avgUCuImpl(int x, int y, int z, int nx, int ny, double* u) 
{
	return (
		u[z * nx * ny + (y - 1) * nx + x] +
		u[z * nx * ny + y * nx + x] +
		u[z * nx * ny + (y - 1) * nx + (x + 1)] +
		u[z * nx * ny + y * nx + (x + 1)] +
		u[(z + 1) * nx * ny + (y - 1) * nx + x] +
		u[(z + 1) * nx * ny + y * nx + x] +
		u[(z + 1) * nx * ny + (y - 1) * nx + (x + 1)] +
		u[(z + 1) * nx * ny + y * nx + (x + 1)]
		) * 0.125;
}

__device__ double avgVCuImpl(int x, int y, int z, int nx, int ny, double* v) 
{
	return (
		v[z * nx * ny + y * nx + (x - 1)] +
		v[z * nx * ny + y * nx + x] +
		v[z * nx * ny + (y + 1) * nx + (x - 1)] +
		v[z * nx * ny + (y + 1) * nx + x] +
		v[(z + 1) * nx * ny + y * nx + (x - 1)] +
		v[(z + 1) * nx * ny + y * nx + x] +
		v[(z + 1) * nx * ny + (y + 1) * nx + (x - 1)] +
		v[(z + 1) * nx * ny + (y + 1) * nx + x]
		) * 0.125;
}

__device__ double avgWCuImpl(int x, int y, int z, int nx, int ny, double* w) 
{
	return (
		w[z * nx * ny + (y - 1) * nx + (x - 1)] +
		w[z * nx * ny + (y - 1) * nx + x] +
		w[z * nx * ny + y * nx + (x - 1)] +
		w[z * nx * ny + y * nx + x] +
		w[(z + 1) * nx * ny + (y - 1) * nx + (x - 1)] +
		w[(z + 1) * nx * ny + (y - 1) * nx + x] +
		w[(z + 1) * nx * ny + y * nx + (x - 1)] +
		w[(z + 1) * nx * ny + y * nx + x]
		) * 0.125;
}

/*
fieldTy: 1.U 2.V 3.S
*/
__device__ double interpolationCuImpl(
	double x, double y, double z, int fieldTy,
	double dxyz, int nx, int ny, int nz, double* arr)
{
	double h1 = 1.0 / dxyz;
	double h2 = 0.5 * dxyz;
	double dx = h2, dy = h2, dz = h2;

	x = myMax(myMin(x, nx * dxyz), dxyz);
	y = myMax(myMin(y, ny * dxyz), dxyz);
	z = myMax(myMin(z, nz * dxyz), dxyz);

	switch (fieldTy)
	{
	case 1: dx = 0; break;
	case 2: dy = 0; break;
	case 3: dz = 0; break;
	default: break;
	}

	int x0 = myMin(floor((x - dx) * h1), nx - 1);
	int y0 = myMin(floor((y - dy) * h1), ny - 1);
	int z0 = myMin(floor((z - dz) * h1), nz - 1);

	double tx = ((x - dx) - x0 * dxyz) * h1;
	double ty = ((y - dy) - y0 * dxyz) * h1;
	double tz = ((z - dz) - z0 * dxyz) * h1;

	int x1 = myMin(x0 + 1, nx - 1);
	int y1 = myMin(y0 + 1, ny - 1);
	int z1 = myMin(z0 + 1, nz - 1);

	double sx = 1.0 - tx;
	double sy = 1.0 - ty;
	double sz = 1.0 - tz;

	return
		sz * (sy * (sx * arr[z0 * nx * ny + y0 * nx + x0] +
			tx * arr[z0 * nx * ny + y0 * nx + x1]) +
			ty * (sx * arr[z0 * nx * ny + y1 * nx + x0] +
				tx * arr[z0 * nx * ny + y1 * nx + x1])) +
		tz * (sy * (sx * arr[z1 * nx * ny + y0 * nx + x0] +
			tx * arr[z1 * nx * ny + y0 * nx + x1]) +
			ty * (sx * arr[z1 * nx * ny + y1 * nx + x0] +
				tx * arr[z1 * nx * ny + y1 * nx + x1]));
}

__global__ void advectVelCuImpl(
	double dt, int nx, int ny, int nz, double dxyz, int threadCnt,
	double* barrier, double* u, double* v, double* w,
	double* newU, double* newV, double* newW)
{
	const int sz = (nx - 2) * (ny - 2) * (nz - 2);
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sz; idx += threadCnt)
	{
		int x = idx % (nx - 2) + 1;
		int y = (idx / (nx - 2)) % (ny - 2) + 1;
		int z = idx / ((nx - 2) * (ny - 2)) + 1;

		const double halfDxyz = 0.5 * dxyz;
		const int baseIdx = z * nx * ny + y * nx + x;

		double avgU = avgUCuImpl(x, y, z, nx, ny, u);
		double avgV = avgVCuImpl(x, y, z, nx, ny, v);
		double avgW = avgWCuImpl(x, y, z, nx, ny, w);

		if (x < nx - 1 && z < nz - 1 &&barrier[baseIdx] != 0 &&barrier[baseIdx - 1] != 0)
		{
			double xPos = x * dxyz;
			double yPos = y * dxyz + halfDxyz;
			double zPos = z * dxyz + halfDxyz;

			xPos -= dt * u[baseIdx];
			yPos -= dt * avgV;
			zPos -= dt * avgW;

			newU[baseIdx] = interpolationCuImpl(xPos, yPos, zPos, 1, dxyz, nx, ny, nz, u);
		}

		if (y < ny - 1 && z < nz - 1 &&barrier[baseIdx] != 0 &&barrier[baseIdx - nx] != 0)
		{
			double xPos = x * dxyz + halfDxyz;
			double yPos = y * dxyz;
			double zPos = z * dxyz + halfDxyz;

			xPos -= dt * avgU;
			yPos -= dt * v[baseIdx];
			zPos -= dt * avgW;

			newV[baseIdx] = interpolationCuImpl(xPos, yPos, zPos, 2, dxyz, nx, ny, nz, v);
		}

		if (z < nz - 1 &&
			barrier[baseIdx] != 0 && barrier[baseIdx - nx * ny] != 0)
		{
			double xPos = x * dxyz + halfDxyz;
			double yPos = y * dxyz + halfDxyz;
			double zPos = z * dxyz;

			xPos -= dt * avgU;
			yPos -= dt * avgV;
			zPos -= dt * w[baseIdx];

			newW[baseIdx] = interpolationCuImpl(xPos, yPos, zPos, 3, dxyz, nx, ny, nz, w);
		}
	}
}

__global__ void advectSmokeCuImpl(
	double dt, int nx, int ny, int nz, double dxyz, int threadCnt,
	double* barrier, double* u, double* v, double* w, double* smoke, double* newSmoke)
{
	unsigned sz = nx * ny * nz;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sz; idx += threadCnt)
	{
		int x = idx % nx;
		int y = (idx / nx) % ny;
		int z = idx / (nx * ny);

		if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) continue;

		double halfDxyz = 0.5 * dxyz;
		if (barrier[z * nx * ny + y * nx + x] != 0)
		{
			double xPos = x * dxyz + halfDxyz;
			double yPos = y * dxyz + halfDxyz;
			double zPos = z * dxyz + halfDxyz;

			int baseIdx = z * nx * ny + y * nx + x;
			double avgU = 0.5 * (u[baseIdx] + u[baseIdx + 1]);
			double avgV = 0.5 * (v[baseIdx] + v[baseIdx + nx]);
			double avgW = 0.5 * (w[baseIdx] + w[baseIdx + nx * ny]);

			xPos -= dt * avgU;
			yPos -= dt * avgV;
			zPos -= dt * avgW;

			newSmoke[baseIdx] = interpolationCuImpl(
				xPos, yPos, zPos, 3, dxyz, nx, ny, nz, smoke);
		}
	}
}

__global__ void clearToZeroCuImpl(int size, int threadCnt, double* arr)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += threadCnt)
	{
		arr[idx] = 0.0;
	}
}

__global__ void initSceneVortexStreetCuImpl(
	int nx, int ny, int nz, double dx, double dy, double dz, int threadCnt,
	double inVel, double pipeMinH, double pipeMaxH,
	double* barrier, double* smoke, double* u)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx * ny * nz; idx += threadCnt)
	{
		int x = idx % nx;
		int y = (idx / nx) % ny;
		int z = idx / (nx * ny);

		if (x >= nx || y >= ny || z >= nz) continue;

		double s = 1.0;
		if (x == 0 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1)
			s = 0.0;
		barrier[z * nx * ny + y * nx + x] = s;

		if (x == 1)
			u[z * nx * ny + y * nx + x] = inVel;

		if (x == 0 && (y > pipeMinH && y < pipeMaxH))
			smoke[z * nx * ny + y * nx + x] = 1.0;
	}
}

__global__ void boundryCondPeriodCuImplXaxis(
	int nx, int ny, int nz, int threadCnt, double* u)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x;idx < ny * nz;idx += threadCnt)
	{
		int y = idx % ny;
		int z = idx / ny;

		if (y >= ny || z >= nz) return;

		int front = z * nx * ny + y * nx;
		int back = z * nx * ny + y * nx + (nx - 1);

		double tmp = u[front];
		u[front] = u[back];
		u[back] = tmp;
	}
}

__global__ void boundryCondPeriodCuImplYaxis(
	int nx, int ny, int nz, int threadCnt, double* u)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x;idx < nx * nz;idx += threadCnt)
	{
		int x = idx % nx;
		int z = idx / nx;

		if (x >= nx || z >= nz) return;

		int bottom = z * nx * ny + x;
		int top = z * nx * ny + (ny - 1) * nx + x;  

		double tmp = u[bottom];
		u[bottom] = u[top];
		u[top] = tmp;
	}
}

__global__ void boundryCondPeriodCuImplZaxis(
	int nx, int ny, int nz, int threadCnt, double* u)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x;idx < nx * ny;idx += threadCnt)
	{
		int x = idx % nx;
		int y = idx / nx;

		if (x >= nx || y >= ny) return;

		int front = y * nx + x;
		int back = (nz - 1) * nx * ny + y * nx + x;

		double tmp = u[front];
		u[front] = u[back];
		u[back] = tmp;
	}
}

void boundryCondPeriodCuImpl(
	int nx, int ny, int nz, double* u, double* v, double* w)
{
	boundryCondPeriodCuImplXaxis << <2, 64 >> > (nx, ny, nz, 64 * 64, u);
	cudaDeviceSynchronize();
	boundryCondPeriodCuImplYaxis << <2, 64 >> > (nx, ny, nz, 64 * 64, v);
	cudaDeviceSynchronize();
	boundryCondPeriodCuImplZaxis << <2, 64 >> > (nx, ny, nz, 64 * 64, w);
	cudaDeviceSynchronize();
}