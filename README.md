# Image_Filters
Using CUDA and Gpu Computation to do batch processing and apply different Image Filters to multiple input jpg or png images

#Features 
**CUDA-Accelerated Processing**: Custom CUDA kernels for fast parallel image processing

**Multiple Filters**: Gaussian blur, Sharpen, DeNoiser

**Batch Processing**: Process entire directories of images

**Multi-Format Support**: Handles both PNG and JPG/JPEG images

**Performance Metrics**: Displays total processing time and per-image averages
# Prerequisites 
### System Requirements
* NVIDIA GPU with CUDA support
* CUDA Toolkit (10.0 or later)
* GCC/G++ compiler
* Linux (tested on Arch Linux)

### Install CUDA 
* Arch Linux:
 sudo pacman -S cuda nvidia nvidia-utils
* Ubuntu:
 sudo apt update
 sudo apt install build-essential nvidia-cuda-toolkit nvidia-driver-535
* Windows:
 1. Install CUDA Toolkit from NVIDIA website.
 2. Install Visual Studio 2022 (with C++ tools).
 3. Verify with:
 nvcc --version
 nvidia-smi

# Setup 
1.  Adjust GPU Architecture
   Edit the makefile and set the correct flag for you gpu

# Building
Linux (Arch/Ubuntu):
 make
 # or
 nvcc -std=c++11 -O3 -arch=sm_75 main.cu -o image_filter
Windows (Developer Command Prompt):
 nvcc -std=c++11 -O3 -arch=sm_75 main.cu -o image_filter.exe

# Usage 
* Linux - ./image_filter <input_directory> <output_directory>
* Windows - image_filter.exe input_images output_images

# License 
Licensed under the MIT License.

# Acknowledgements 
- STB libraries by Sean Barrett
- NVIDIA CUDA Toolkit and documentation
