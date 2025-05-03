#include "Fluid.cuh"
#include "FluidKernel.cuh"

Scene scene;

v1d& Fluid::getU() { return u; }

v1d& Fluid::getV() { return v; }

v1d& Fluid::getW() { return w; }

v1d& Fluid::getSmoke() { return smoke; }

v1d& Fluid::getBrrier() { return barrier; }

v1d& Fluid::getPressure() { return pressure; }

int Fluid::getNX() { return nx; }

int Fluid::getNY() { return ny; }

int Fluid::getNZ()
{
	return nz;
}

double Fluid::getdxy() { return dxyz; }

Fluid::Fluid(double density, int NX, int NY,int NZ, double h)
{
	this->dxyz = h;
	this->density = density;
	nx = NX + 2, ny = NY + 2, nz = NZ + 2;

	smoke = newSmoke = v1d(nz * ny * nx);
	w = u = v = newW = newU = newV = barrier = pressure = v1d(nz * ny * nx);
}

void Fluid::integrate(double dt, double gravity)
{
	employGravityCuImpl(dt, gravity, v);
}

void Fluid::advectVel(double dt)
{
	newU = u, newV = v, newW = w;
	advectVelCuImpl << <256, 64 >> > (
		dt, nx, ny, nz, dxyz, 256 * 64, rawPtr4(barrier, u, v, w, ), rawPtr3(newU, newV, newW));
	cudaDeviceSynchronize();
	u = newU, v = newV, w = newW;
}

void Fluid::advectSmoke(double dt)
{
	newSmoke = smoke;
	advectSmokeCuImpl << <256, 64 >> > (
		dt, nx, ny, nz, dxyz, 256 * 64, rawPtr4(barrier, u, v, w), rawPtr2(smoke, newSmoke));
	cudaDeviceSynchronize();
	smoke = newSmoke;
}

void Fluid::simulate(double dt, double gravity, int numIters)
{
	integrate(dt, gravity);

	clearToZeroCuImpl << <32, 128 >> > (
		nx * ny * nz, 32 * 128, rawPtr1(pressure));
	cudaDeviceSynchronize();

	Solver::PCG_GSRB(numIters, dt, *this);

	boundryCondPeriod();
	advectVel(dt);
	advectSmoke(dt);
}

void Fluid::Solver::GaussSedielRB(int myMaxIter, double dt, Fluid& f)
{
	const double pdxydt = f.density * f.dxyz / dt * 1.9;
	GaussSedielRBCu(
		myMaxIter, dt, f.nx, f.ny, f.nz, pdxydt,
		rawPtr5(f.barrier, f.u, f.v, f.w, f.pressure));
}

void Fluid::Solver::ConjugateGradient(int myMaxIter, double dt, Fluid& f)
{
	ConjugateGradientCu(
		myMaxIter, dt, f.nx, f.ny, f.nz, f.density * f.dxyz / dt,
		f.barrier, f.u, f.v, f.w, f.pressure);
}


void Fluid::Solver::PCG_GSRB(int myMaxIter, double dt, Fluid& f)
{
    GaussSedielRB(myMaxIter * 0.2, dt, f);
    ConjugateGradient(myMaxIter * 0.8, dt, f);
}