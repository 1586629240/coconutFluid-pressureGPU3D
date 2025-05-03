#pragma once

#include "Fluid.cuh"
#include "covert2vdb.h"

void setObstacle(double x, double y, bool reset, double t);

void initScenePaint(size_t nxyz);

void initSceneVortexStreet(size_t nxyz);

void simulate();

void display(Fluid& fluid);

void mouseEvent(Fluid& cube);

void exportToFile(Fluid& fluid);
