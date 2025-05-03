# coconutFluid-pressureGPU3D

the name is too long...

# Description

This project is use SIMPLE-like algorithm to solving the N-S equation in 3D space. We implement the 3D vortex street problem.

Employing the Semi Lagrangian scheme to push the smoke and velocity move under the presure.

Different with another project. In pressureGPU3D, we employ CUDA to accelerate the calculate speed. Further more, the pressureGPU3D will export VDB volume file sequence to floder G:/smoke. The VDB file can be opened with C4D, blender, 3DMax and other 3d creative software. You can open VDB file, edit it and then rendering the whole sequence!

# Tips

The project can be complie in release mode, but if switch to debug mode, you may be need to re-configure the openvdb and Intel-TBB library in linker option.

When edit source code, please make sure that the simulation When edit source code, please make sure that the simulation is meet the CFL condition is meet the CFL condition requirement, or the simulation will break down.
 
 The export path can be edit to other. Especially when your computer don't have the "G:/"

# How to compile

1. Only can run on windows. Compiling on windows is strongly recommended. Further more, your computer must have a NVIDIA graphics card.

2. Please download VisualStudio2022.

3. Download the cuda sdk on NVIDIA website, the recommended version is 12.6
   
4. Open sln file in VisualStudio2022, and then compile. May need to change the architecture "sm_86" to other
