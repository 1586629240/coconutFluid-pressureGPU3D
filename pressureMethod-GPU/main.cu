#include "misc.cuh"

int run()
{	
	vdbInit();
	initSceneVortexStreet(200);
	for (unsigned counter = 0; counter < 10000;)
	{
		simulate();
		exportToFile(*scene.fluid);
		printf("Progress: %lf\n", counter++ / 10000.);
	}

	delete scene.fluid;
	return 0;
}

int profile()
{
	initSceneVortexStreet(200);
	simulate();
	delete scene.fluid;
	return 0;
}

int main(int argc, char* argv[])
{
	if (argc > 1)
		return profile();
	else
		return run();
}