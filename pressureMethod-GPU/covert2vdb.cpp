#include "covert2vdb.h"

#include <openvdb\openvdb.h>

void vdbInit()
{
	openvdb::initialize();
}

void convert2vdb(double* data, unsigned nx, unsigned ny, unsigned nz, std::string filename)
{
	openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
	grid->setName("smokeSimu");
	auto accessor = grid->getAccessor();

	for (int i = 0; i < nz; ++i)
		for (int j = 0; j < ny; ++j)
			for (int k = 0; k < nx; ++k)
				accessor.setValue(openvdb::Coord(i, j, k), data[i * ny * nx + j * nx + k]);

	openvdb::io::File file(filename);
	file.write({ grid });
	file.close();
}