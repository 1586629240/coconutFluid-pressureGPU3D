#include "FluidKernel.cuh"
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>

__global__ void updateVel0CuImpl(
    int nx, int ny, int nz, int threadCnt,
    double* barrier, double* u, double* v, double* w, double* p)
{
    int total_xy = (nx - 2) * (ny - 2);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx * ny * nz; idx += threadCnt)
    {
        int z = idx / total_xy;
        int remainder = idx % total_xy;
        int y = remainder / (nx - 2) + 1;
        int x = remainder % (nx - 2) + 1;
        z += 1;

        if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) continue;
        if (barrier[z * ny * nx + y * nx + x] == 0) continue;

        double sx0 = barrier[z * ny * nx + y * nx + (x - 1)];
        double sy0 = barrier[z * ny * nx + (y - 1) * nx + x];
        double sz0 = barrier[(z - 1) * ny * nx + y * nx + x];

        if (sx0 + sy0 + sz0 == 0) continue;

        double p_val = p[z * ny * nx + y * nx + x];
        u[z * ny * nx + y * nx + x] -= sx0 * p_val;
        v[z * ny * nx + y * nx + x] -= sy0 * p_val;
        w[z * ny * nx + y * nx + x] -= sz0 * p_val;
    }
}

__global__ void updateVel1CuImpl(
    int nx, int ny, int nz, int threadCnt,
    double* barrier, double* u, double* v, double* w, double* p)
{
    int total_xy = (nx - 2) * (ny - 2);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx * ny * nz; idx += threadCnt)
    {
        int z = idx / total_xy;
        int remainder = idx % total_xy;
        int y = remainder / (nx - 2) + 1;
        int x = remainder % (nx - 2) + 1;
        z += 1;

        if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) continue;
        if (barrier[z * ny * nx + y * nx + x] == 0) continue;

        double sx1 = barrier[z * ny * nx + y * nx + (x + 1)];
        double sy1 = barrier[z * ny * nx + (y + 1) * nx + x];
        double sz1 = barrier[(z + 1) * ny * nx + y * nx + x];

        if (sx1 + sy1 + sz1 == 0) continue;

        double p_val = p[z * ny * nx + y * nx + x];
        u[z * ny * nx + y * nx + x + 1] += sx1 * p_val;
        v[z * ny * nx + (y + 1) * nx + x] += sy1 * p_val;
        w[(z + 1) * ny * nx + y * nx + x] += sz1 * p_val;
    }
}

__global__ void ComputeBAndSvalKernel(
    int nx, int ny, int nz, int threadCnt,
    double* barrier, double* u, double* v, double* w, double* b, double* sval)
{
    int total_xy = (nx - 2) * (ny - 2);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx * ny * nz; idx += threadCnt)
    {
        int z = idx / total_xy;
        int remainder = idx % total_xy;
        int y = remainder / (nx - 2) + 1;
        int x = remainder % (nx - 2) + 1;
        z += 1;

        if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) continue;

        double sx0 = barrier[z * ny * nx + y * nx + (x - 1)];
        double sx1 = barrier[z * ny * nx + y * nx + (x + 1)];
        double sy0 = barrier[z * ny * nx + (y - 1) * nx + x];
        double sy1 = barrier[z * ny * nx + (y + 1) * nx + x];
        double sz0 = barrier[(z - 1) * ny * nx + y * nx + x];
        double sz1 = barrier[(z + 1) * ny * nx + y * nx + x];

        double s = sx0 + sx1 + sy0 + sy1 + sz0 + sz1;
        sval[z * ny * nx + y * nx + x] = s;

        if (s == 0) {
            b[z * ny * nx + y * nx + x] = 0.0;
            continue;
        }

        double div =
            u[z * ny * nx + y * nx + (x + 1)] - u[z * ny * nx + y * nx + x] +
            v[z * ny * nx + (y + 1) * nx + x] - v[z * ny * nx + y * nx + x] +
            w[(z + 1) * ny * nx + y * nx + x] - w[z * ny * nx + y * nx + x];

        b[z * ny * nx + y * nx + x] = -div;
    }
}

__global__ void MatrixVectorMultiplyKernel(
    int nx, int ny, int nz, int threadCnt,
    double* barrier, double* sval, double* vec, double* result)
{
    int total_xy = (nx - 2) * (ny - 2);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx * ny * nz; idx += threadCnt)
    {
        int z = idx / total_xy;
        int remainder = idx % total_xy;
        int y = remainder / (nx - 2) + 1;
        int x = remainder % (nx - 2) + 1;
        z += 1;

        if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) continue;

        double s = sval[z * ny * nx + y * nx + x];
        double sum =
            vec[z * ny * nx + y * nx + (x - 1)] * barrier[z * ny * nx + y * nx + (x - 1)] +
            vec[z * ny * nx + y * nx + (x + 1)] * barrier[z * ny * nx + y * nx + (x + 1)] +
            vec[z * ny * nx + (y - 1) * nx + x] * barrier[z * ny * nx + (y - 1) * nx + x] +
            vec[z * ny * nx + (y + 1) * nx + x] * barrier[z * ny * nx + (y + 1) * nx + x] +
            vec[(z - 1) * ny * nx + y * nx + x] * barrier[(z - 1) * ny * nx + y * nx + x] +
            vec[(z + 1) * ny * nx + y * nx + x] * barrier[(z + 1) * ny * nx + y * nx + x];

        result[z * ny * nx + y * nx + x] = s * vec[z * ny * nx + y * nx + x] - sum * (s != 0);
    }
}

__global__ void dot3MatKernel(
    unsigned size, int threadCnt,
    double* a, double* b, double* c, double* res)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += threadCnt)
        res[idx] = a[idx] * b[idx] * c[idx];
}

__global__ void rudeceKernel(const double* input, double* result, int N)
{
    __shared__ double smem[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[threadIdx.x] = (tid < N) ? input[tid] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)atomicAdd(result, smem[0]);
}

double myReduce(const thrust::device_vector <double>& data)
{
    thrust::device_vector <double> result(1, 0);
    const int block_size = 256;
    int grid_size = (data.size() + block_size - 1) / block_size;

    rudeceKernel << <grid_size, block_size >> > (rawPtr2(data, result), data.size());
    cudaDeviceSynchronize();
    return result[0];
}

double dot3MatCu(v1d& a, v1d& b, v1d& c)
{
    static auto tmp = a;

    dot3MatKernel << <128, 128 >> > (a.size(), 128 * 128, rawPtr4(a, b, c, tmp));
    cudaDeviceSynchronize();
    return myReduce(tmp);
}

__global__ void xAddcMulyCuKernel(
    unsigned size, int threadCnt,
    double* a, double* b, double* res, double c)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += threadCnt)
        res[idx] = a[idx] + c * b[idx];
}

void xAddcMulyCu(v1d& a, v1d& b, bool resPos, double c)
{
    if (resPos == 0)
        xAddcMulyCuKernel << <128, 128 >> > (a.size(), 128 * 128, rawPtr3(a, b, a), c);
    else
        xAddcMulyCuKernel << <128, 128 >> > (a.size(), 128 * 128, rawPtr3(a, b, b), c);
    cudaDeviceSynchronize();
}

void ConjugateGradientCu(
    int maxIter, double dt, int nx, int ny, int nz, double pdxydt,
    v1d& barrier, v1d& u, v1d& v, v1d& w, v1d& pressure)
{
    thrust::device_vector<double> d_b(nz * ny * nx, 0.0);
    thrust::device_vector<double> d_sval(nz * ny * nx, 0.0);

    ComputeBAndSvalKernel << <128, 128 >> > (nx, ny, nz, 128 * 128,
        rawPtr3(barrier, u, v), rawPtr3(w, d_b, d_sval));
    cudaDeviceSynchronize();

    thrust::device_vector<double> d_r = d_b;
    thrust::device_vector<double> d_d = d_b;
    thrust::device_vector<double> d_p(nz * ny * nx, 0.0);
    thrust::device_vector<double> d_Ad(nz * ny * nx, 0.0);

    double rr_old = dot3MatCu(d_r, d_r, barrier);
    if (rr_old < 1e-6) return;

    for (int iter = 0; iter < maxIter; ++iter) {
        MatrixVectorMultiplyKernel << <128, 128 >> > (nx, ny, nz, 128 * 128,
            rawPtr4(barrier, d_sval, d_d, d_Ad));
        cudaDeviceSynchronize();

        double d_dot_Ad = dot3MatCu(d_d, d_Ad, barrier);
        double alpha = rr_old / (d_dot_Ad + 1e-6);

        xAddcMulyCu(d_p, d_d, 0, alpha);
        xAddcMulyCu(d_r, d_Ad, 0, -alpha);

        double rr_new = dot3MatCu(d_r, d_r, barrier);
        if (std::sqrt(rr_new) < 1e-6) break;

        double beta = rr_new / rr_old;
        xAddcMulyCu(d_r, d_d, 1, beta);
        rr_old = rr_new;
    }

    xAddcMulyCu(pressure, d_p, 0, pdxydt);

    updateVel0CuImpl << <128, 128 >> > (nx, ny, nz, 128 * 128,
        rawPtr5(barrier, u, v, w, d_p));
    cudaDeviceSynchronize();
    updateVel1CuImpl << <128, 128 >> > (nx, ny, nz, 128 * 128,
        rawPtr5(barrier, u, v, w, d_p));
    cudaDeviceSynchronize();
}