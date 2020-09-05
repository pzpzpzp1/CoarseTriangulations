# Coarse Triangulations

Given an input image, compute a coarse triangulation approximation of the image. Supported approximations: piecewise constant, linear, quadratic per triangle. An initial triangulation on the image is provided by the TRIM algorithm, and then a gradient descent mesh moving method is used to further align the triangular mesh to the image to minimize approximation error.

## Credits

Polyscope: www.polyscope.run

TRIM: C. P. Yung, G. P. T. Choi, K. Chen and L. M. Lui, "Efficient feature-based image registration by mapping sparsified surfaces." Journal of Visual Communication and Image Representation, 55, pp. 561-571, 2018.

Saliency Toolbox: Dirk Walther and Christof Koch (2006), Modeling attention to salient proto-objects. Neural Networks 19, 1395-1407. 

## Requirements

For image reading, `libjpeg` and `libpng` with `libz` are needed, and `X11` is used for testing. For speed purposes, an Nvidia GPU is needed with CUDA support. Other installations not included in this repo include MATLAB and `ffmpeg` (for Linux; this is not strictly necessary and is just used for output purposes).

## Compiling

On Linux, compiling can be done by
```
mkdir build
cd build
cmake ..
make
```

This will compile the main file as well as the tests, which can be found in the `bin` directory of the build.

## Running Triangulations

Put an image (for the sake of example, `image.jpg`) to be triangulated in the `images` directory. Then from the `build` directory, run ```./CoarseTriangulation image.jpg``` This will output a screenshot of the initial triangulation, the final coarse triangulation, a video of all intermediate steps, and graphs of the energy function and total step over time. All these can be found in the `outputs` directory. Note that running the program multiple times will overwrite these files, which may require root access.

For the sake of convenience, an image cropper can also be found in the `outputs` directory. This will remove the white Polyscope background. Compile this executable by
```
mkdir build && cd build
cmake ..
make
```
from within the `outputs` directory. Then run ```build/crop relative_path_to_image output_file_name``` from the `outputs` directory, which will output a file named `output_file_name` (default `cropped.jpeg`). Note that if the image input to the triangulation already has a white background, this image cropper may remove part of that background as well. To fix this, simply convert the triangulation output to a png file before running the cropper.
