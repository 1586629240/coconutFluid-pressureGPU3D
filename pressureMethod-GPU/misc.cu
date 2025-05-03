#include "misc.cuh"
#include "FluidKernel.cuh"
#include <thrust/host_vector.h>
#include <fstream>
#include <string>

void setObstacle(double x, double y, bool reset, double t = 0)
{
	auto f = scene.fluid;
	double oldX = scene.ballX, oldY = scene.ballY;
	scene.ballX = x, scene.ballY = y;

	if (x <= scene.ballR || x >= f->getNX() - scene.ballR)return;
	if (y <= scene.ballR || y >= f->getNY() - scene.ballR)return;

	double vx = (!reset) * ((x - oldX) / scene.dt);
	double vy = (!reset) * ((y - oldY) / scene.dt);

	vx = max(min(vx, 2.0), -2.0);
	vy = max(min(vy, 2.0), -2.0);

	setObstacleCuImpl << <16, 128 >> > (
		vx, vy, f->getNX(), f->getNY(), f->getNZ(), f->getdxy(), 16 * 128,
		x, y, scene.ballR * scene.ballR, t,
		rawPtr5(f->getBrrier(), f->getU(), f->getV(), f->getW(), f->getSmoke()));
	cudaDeviceSynchronize();
}

void initScenePaint(size_t nxyz)
{
	scene.dt = 0.001;
	scene.ballR = 0.05;

	double dxdy = 1.0 / nxyz;
	int nx = nxyz, ny = nxyz, nz = nxyz;

	double density = 1000.0;
	scene.fluid = new Fluid(density, nx, ny, nz, dxdy);

	scene.gravity = 0;
	scene.enableMouse = true;
}

void initSceneVortexStreet(size_t nxyz)
{
	scene.dt = 0.01;
	scene.ballR = 0.05;

	double dxyz = 1.0 / nxyz;
	int nx = nxyz, ny = nxyz, nz = nxyz;

	double density = 100000.0;
	auto f = scene.fluid = new Fluid(density, nx, ny, nz, dxyz);

	double inVel = 2;
	double pipeH = 0.05 * f->getNY();
	double pipeMinH = floor(0.5 * f->getNY() - 0.5 * pipeH);
	double pipeMaxH = floor(0.5 * f->getNY() + 0.5 * pipeH);

	initSceneVortexStreetCuImpl << <16, 128 >> > (
		nx + 2, ny + 2, nz + 2, dxyz, dxyz, dxyz, 16 * 128,
		inVel, pipeMinH, pipeMaxH, rawPtr3(f->getBrrier(), f->getSmoke(), f->getU()));
	cudaDeviceSynchronize();

	setObstacle(0.2, 0.5, true);
	scene.gravity = 0.0;
	scene.enableMouse = true;
}

void simulate()
{
	for (int i = 0; i < 1; i++)
		scene.fluid->simulate(scene.dt, scene.gravity, 50);
}

void display(Fluid& fluid) 
{
	int N = fluid.getNY();
	auto buf = GetImageBuffer();

	int nx = fluid.getNX();
	thrust::host_vector<double> smoke = fluid.getSmoke();
	unsigned base = fluid.getNY() * nx;

	for (int y = 0; y < 480; y++) 
	{
		for (int x = 0; x < 640; x++) 
		{
			int dx = static_cast<int>((x / 640.0f) * N);
			int dy = static_cast<int>((y / 480.0f) * N);
			float d = smoke[100 * base + dy * nx + dx];
			buf[y * 640 + x] = HSVtoRGB(d * 360, 1, 1);
			//buf[y * 640 + x] = RGB(d * 255, d * 255, d * 255);
		}
	}
	FlushBatchDraw();
}

void mouseEvent(Fluid& cube)
{
	if (scene.enableMouse == false) return;

	static bool drawing = false;
	static double t = 0;
	ExMessage m;
	if (peekmessage(&m))
	{
		switch (m.message)
		{
		case WM_LBUTTONDOWN: drawing = true; break;
		case WM_LBUTTONUP:   drawing = false; break;
		}
	}

	if (drawing && m.message == WM_MOUSEMOVE)
	{
		double x = m.x / 640.;
		double y = m.y / 480.;
		setObstacle(x, y, false, t);
		t += 0.001;
	}
}

void exportToFile(Fluid& fluid)
{
	static int id = 0;
	thrust::host_vector<double> smoke = fluid.getSmoke();

	std::string filename = "G:\\smoke\\" + std::to_string(id++) + ".vdb";

	convert2vdb(
		thrust::raw_pointer_cast(smoke.data()), fluid.getNX(), fluid.getNY(), fluid.getNZ(),
		filename.c_str());
}