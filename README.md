# Coarse Triangulations

Given an input image, compute a coarse triangulation approximation of the image. Supported approximations: piecewise constant, linear, quadratic per triangle. 
An initial triangulation on the image can be provided, after which gradient descent mesh moving method is applied to further align the triangular mesh to the image.
A saliency map can be provided which influences what part of the image is most important vs less important.

## Credits

Original implementation of the image triangulations method was written by Wanlin Li for Linux. This version is only accessible to members of MIT:
 https://github.mit.edu/liwanlin/image-triangulation

Polyscope: www.polyscope.run

lodepng: https://github.com/lvandeve/lodepng
(Thank god this exists. libpng, and zlib support for windows is rather lacking in comparison.)

## Requirements

Nvidia GPU with CUDA toolkits

Visual Studio

## Compiling

Make sure you have the dependencies lodepng and polyscope in the deps/ directory. These are git submodules.
Build polyscope using cmake to generate a polyscope sln file.
Open the redo.sln file using Visual Studio (not the polyscope sln file).
Hit the green arrow.

## Running Triangulations

Input/Output format is still under development.
Roughly speaking, you can input a png filepath via commandline.
I won't detail this step much since it's subject to rapid oncoming change.
