#include "FluidKernel.cuh"


__device__ void GaussSedielKernelCuImpl(
    int x, int y, int z, int nx, int ny, int nz, double pdxydt,
    double* barrier, double* u, double* v, double* w, double* pre)
{
    const int idx = z * nx * ny + y * nx + x;
    if (barrier[idx] == 0) return; 

    const double sx0 = (x > 0) ? barrier[idx - 1] : 0;
    const double sx1 = (x < nx - 1) ? barrier[idx + 1] : 0;
    const double sy0 = (y > 0) ? barrier[idx - nx] : 0;
    const double sy1 = (y < ny - 1) ? barrier[idx + nx] : 0;
    const double sz0 = (z > 0) ? barrier[idx - nx * ny] : 0;
    const double sz1 = (z < nz - 1) ? barrier[idx + nx * ny] : 0;

    const double s = sx0 + sx1 + sy0 + sy1 + sz0 + sz1;
    if (s == 0) return;

    const double div =
        (u[idx + 1] - u[idx]) + (v[idx + nx] - v[idx]) + (w[idx + nx * ny] - w[idx]);

    const double p = -div / s;
    pre[idx] += pdxydt * p;

    if (x > 0)    u[idx] -= sx0 * p;
    if (x < nx - 1) u[idx + 1] += sx1 * p;
    if (y > 0)    v[idx] -= sy0 * p;
    if (y < ny - 1) v[idx + nx] += sy1 * p;
    if (z > 0)    w[idx] -= sz0 * p;
    if (z < nz - 1) w[idx + nx * ny] += sz1 * p;
}

thrust::device_vector<unsigned> redIdx3D;
thrust::device_vector<unsigned> blackIdx3D;

void initRedBlackIdx3D(int nx, int ny, int nz) 
{
    if (!redIdx3D.empty()) return;

    std::vector<unsigned> redHost, blackHost;
    const int total = (nx - 2) * (ny - 2) * (nz - 2);
    redHost.reserve(total / 2 + 1);
    blackHost.reserve(total / 2 + 1);

    for (int z = 1; z < nz - 1; ++z) 
    {
        for (int y = 1; y < ny - 1; ++y) 
        {
            for (int x = 1; x < nx - 1; ++x) 
            {
                const int coordSum = x + y + z;
                const int flatIdx = (z - 1) * (nx - 2) * (ny - 2) + (y - 1) * (nx - 2) + (x - 1);
                (coordSum % 2 == 0) ? redHost.push_back(flatIdx)
                    : blackHost.push_back(flatIdx);
            }
        }
    }

    redIdx3D = redHost;
    blackIdx3D = blackHost;
}


__global__ void GaussSedielCuImpl(
    double dt, int nx, int ny, int nz, int threadCnt,
    double pdxydt, unsigned* gridIdx, unsigned size,
    double* barrier, double* u, double* v, double* w, double* pre)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += threadCnt) 
    {
        const int flatIdx = gridIdx[idx];

        const int x = (flatIdx % ((nx - 2) * (ny - 2))) % (nx - 2) + 1;
        const int y = (flatIdx % ((nx - 2) * (ny - 2))) / (nx - 2) + 1;
        const int z = flatIdx / ((nx - 2) * (ny - 2)) + 1;

        GaussSedielKernelCuImpl(
            x, y, z, nx, ny, nz, pdxydt, barrier, u, v, w, pre);
    }
}

__host__ void GaussSedielRBCu(
    int iter, double dt, int nx, int ny, int nz, double pdxydt,
    double* barrier, double* u, double* v, double* w, double* pre)
{
    initRedBlackIdx3D(nx, ny, nz);
    const int blockSize = 128;
    const int gridSize = (redIdx3D.size() + blockSize - 1) / blockSize;

    for (int i = 0; i < iter; ++i) 
    {
        GaussSedielCuImpl << <gridSize, blockSize >> > (
            dt, nx, ny, nz, blockSize * gridSize, pdxydt,
            thrust::raw_pointer_cast(redIdx3D.data()), redIdx3D.size(),
            barrier, u, v, w, pre);
        cudaDeviceSynchronize();

        GaussSedielCuImpl << <gridSize, blockSize >> > (
            dt, nx, ny, nz, blockSize * gridSize, pdxydt,
            thrust::raw_pointer_cast(blackIdx3D.data()), blackIdx3D.size(),
            barrier, u, v, w, pre);
        cudaDeviceSynchronize();
    }
}