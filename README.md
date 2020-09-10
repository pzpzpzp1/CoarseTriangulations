# Coarse Triangulations

Given an input image, compute a coarse triangulation approximation of the image. Supported approximations: piecewise constant, and linear. 

## Credits

Original implementation of the image triangulations method was written by Wanlin Li for Linux/Mac. This version is only accessible to members of MIT:
 https://github.mit.edu/liwanlin/image-triangulation

Polyscope: www.polyscope.run

lodepng: https://github.com/lvandeve/lodepng
(Thank god this exists. libpng, and zlib support for windows is rather lacking in comparison.)

## Requirements

Nvidia GPU with appropriate CUDA toolkits. 

GPU must support at least 512 threads per block in debug mode (though you can change this to be lower if necessary in parallelInt.cuh 'threads1D')

Visual Studio to build and run

## Compiling

Make sure you have the dependencies lodepng and polyscope in the deps/ directory. These are git submodules.

Build polyscope using cmake to generate a polyscope sln file.

Open the redo.sln file using Visual Studio (not the polyscope sln file).

Hit the green arrow.

## Running Triangulations
Takes three commandline arguments:

./redo.exe filepath degree dx

filepath is a path to the png input image. Defaults to ../images/toucan.png
degree should be 0 or 1 indicating constant or linear color per triangle. Defaults to 0.
dx indicates pixels per triangle width for the initial triangulation. Defaults to 50.

Note that the visual studio solution is set to input commandline arguments when debugging corresponding to:
./redo.exe ../images/toucan.png 1 100



