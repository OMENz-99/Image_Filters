#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <chrono>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_SIZE 16
#define MAX_KERNEL_SIZE 15
#define SHARPEN_KERNEL_SIZE 3

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Dynamic Gaussian blur kernel
__constant__ float d_gaussianKernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
__constant__ int d_kernelSize;

// Sharpen kernel (3x3)
__constant__ float d_sharpenKernel[SHARPEN_KERNEL_SIZE * SHARPEN_KERNEL_SIZE];

// CUDA kernel for Gaussian blur with variable kernel size
__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, 
                                   int width, int height, int channels, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int radius = kernelSize / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                
                int kernelIdx = (ky + radius) * kernelSize + (kx + radius);
                int imageIdx = (iy * width + ix) * channels + c;
                
                sum += input[imageIdx] * d_gaussianKernel[kernelIdx];
            }
        }
        
        output[(y * width + x) * channels + c] = (unsigned char)min(max(sum, 0.0f), 255.0f);
    }
}

// CUDA kernel for Sharpen filter
__global__ void sharpenKernel(unsigned char* input, unsigned char* output, 
                              int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int radius = SHARPEN_KERNEL_SIZE / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                
                int kernelIdx = (ky + radius) * SHARPEN_KERNEL_SIZE + (kx + radius);
                int imageIdx = (iy * width + ix) * channels + c;
                
                sum += input[imageIdx] * d_sharpenKernel[kernelIdx];
            }
        }
        
        output[(y * width + x) * channels + c] = (unsigned char)min(max(sum, 0.0f), 255.0f);
    }
}

// Non-Local Means Denoising kernel
__global__ void nlmDenoiseKernel(unsigned char* input, unsigned char* output,
                                 int width, int height, int channels,
                                 int searchWindow, int patchSize, float h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int patchRadius = patchSize / 2;
    int searchRadius = searchWindow / 2;
    
    for (int c = 0; c < channels; c++) {
        float weightSum = 0.0f;
        float pixelSum = 0.0f;
        
        // Search window around current pixel
        for (int sy = -searchRadius; sy <= searchRadius; sy++) {
            for (int sx = -searchRadius; sx <= searchRadius; sx++) {
                int searchX = min(max(x + sx, 0), width - 1);
                int searchY = min(max(y + sy, 0), height - 1);
                
                // Compare patches
                float patchDist = 0.0f;
                int patchCount = 0;
                
                for (int py = -patchRadius; py <= patchRadius; py++) {
                    for (int px = -patchRadius; px <= patchRadius; px++) {
                        int px1 = min(max(x + px, 0), width - 1);
                        int py1 = min(max(y + py, 0), height - 1);
                        int px2 = min(max(searchX + px, 0), width - 1);
                        int py2 = min(max(searchY + py, 0), height - 1);
                        
                        int idx1 = (py1 * width + px1) * channels + c;
                        int idx2 = (py2 * width + px2) * channels + c;
                        
                        float diff = (float)input[idx1] - (float)input[idx2];
                        patchDist += diff * diff;
                        patchCount++;
                    }
                }
                
                patchDist /= patchCount;
                
                // Calculate weight using Gaussian function
                float weight = expf(-patchDist / (h * h));
                
                int searchIdx = (searchY * width + searchX) * channels + c;
                pixelSum += weight * input[searchIdx];
                weightSum += weight;
            }
        }
        
        output[(y * width + x) * channels + c] = 
            (unsigned char)min(max(pixelSum / weightSum, 0.0f), 255.0f);
    }
}

// Bilateral filter kernel (faster alternative denoiser)
__global__ void bilateralFilterKernel(unsigned char* input, unsigned char* output,
                                      int width, int height, int channels,
                                      int windowSize, float sigmaSpatial, float sigmaRange) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int radius = windowSize / 2;
    float spatialCoeff = -0.5f / (sigmaSpatial * sigmaSpatial);
    float rangeCoeff = -0.5f / (sigmaRange * sigmaRange);
    
    for (int c = 0; c < channels; c++) {
        float weightSum = 0.0f;
        float pixelSum = 0.0f;
        
        int centerIdx = (y * width + x) * channels + c;
        float centerValue = (float)input[centerIdx];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                
                int idx = (iy * width + ix) * channels + c;
                float pixelValue = (float)input[idx];
                
                // Spatial weight (Gaussian based on distance)
                float spatialDist = kx * kx + ky * ky;
                float spatialWeight = expf(spatialDist * spatialCoeff);
                
                // Range weight (Gaussian based on intensity difference)
                float rangeDist = (centerValue - pixelValue) * (centerValue - pixelValue);
                float rangeWeight = expf(rangeDist * rangeCoeff);
                
                float weight = spatialWeight * rangeWeight;
                
                pixelSum += weight * pixelValue;
                weightSum += weight;
            }
        }
        
        output[(y * width + x) * channels + c] = 
            (unsigned char)min(max(pixelSum / weightSum, 0.0f), 255.0f);
    }
}

// Generate Gaussian kernel with given size and sigma
void generateGaussianKernel(float* kernel, int size, float sigma) {
    int radius = size / 2;
    float sum = 0.0f;
    
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float value = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel[(y + radius) * size + (x + radius)] = value;
            sum += value;
        }
    }
    
    // Normalize
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

void initializeKernels(int blurIntensity) {
    // Determine kernel size and sigma based on blur intensity
    int kernelSize;
    float sigma;
    
    switch(blurIntensity) {
        case 1: // Light blur
            kernelSize = 5;
            sigma = 1.0f;
            break;
        case 2: // Medium blur
            kernelSize = 7;
            sigma = 2.0f;
            break;
        case 3: // Strong blur
            kernelSize = 9;
            sigma = 3.0f;
            break;
        case 4: // Very strong blur
            kernelSize = 11;
            sigma = 4.0f;
            break;
        case 5: // Extreme blur
            kernelSize = 15;
            sigma = 5.0f;
            break;
        default:
            kernelSize = 7;
            sigma = 2.0f;
    }
    
    printf("  Blur settings: Kernel size = %dx%d, Sigma = %.1f\n", 
           kernelSize, kernelSize, sigma);
    
    // Generate Gaussian kernel
    float* h_gaussianKernel = (float*)malloc(kernelSize * kernelSize * sizeof(float));
    generateGaussianKernel(h_gaussianKernel, kernelSize, sigma);
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_gaussianKernel, h_gaussianKernel, 
                                   kernelSize * kernelSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernelSize, &kernelSize, sizeof(int)));
    
    free(h_gaussianKernel);
    
    // Sharpen kernel (3x3) - unchanged
    float h_sharpenKernel[SHARPEN_KERNEL_SIZE * SHARPEN_KERNEL_SIZE] = {
         0.0f, -1.0f,  0.0f,
        -1.0f,  5.0f, -1.0f,
         0.0f, -1.0f,  0.0f
    };
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_sharpenKernel, h_sharpenKernel, 
                                   sizeof(h_sharpenKernel)));
}

void processImage(const char* inputPath, const char* outputPath, 
                  int filterType, int intensity) {
    int width, height, channels;
    unsigned char* h_input = stbi_load(inputPath, &width, &height, &channels, 0);
    
    if (!h_input) {
        fprintf(stderr, "Failed to load image: %s\n", inputPath);
        return;
    }
    
    printf("  Processing: %s (%dx%d, %d channels)\n", inputPath, width, height, channels);
    
    size_t imageSize = width * height * channels * sizeof(unsigned char);
    
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    if (filterType == 1) { // Gaussian Blur
        int kernelSize;
        CUDA_CHECK(cudaMemcpyFromSymbol(&kernelSize, d_kernelSize, sizeof(int)));
        gaussianBlurKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, 
                                                   channels, kernelSize);
    } else if (filterType == 2) { // Sharpen
        sharpenKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels);
    } else if (filterType == 3) { // Bilateral Denoise (Fast)
        int windowSize;
        float sigmaSpatial, sigmaRange;
        
        switch(intensity) {
            case 1: // Light
                windowSize = 5;
                sigmaSpatial = 3.0f;
                sigmaRange = 50.0f;
                break;
            case 2: // Medium
                windowSize = 7;
                sigmaSpatial = 5.0f;
                sigmaRange = 75.0f;
                break;
            case 3: // Strong
                windowSize = 9;
                sigmaSpatial = 7.0f;
                sigmaRange = 100.0f;
                break;
            default:
                windowSize = 7;
                sigmaSpatial = 5.0f;
                sigmaRange = 75.0f;
        }
        
        bilateralFilterKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height,
                                                      channels, windowSize, 
                                                      sigmaSpatial, sigmaRange);
    } else if (filterType == 4) { // NLM Denoise (Quality)
        int searchWindow, patchSize;
        float h;
        
        switch(intensity) {
            case 1: // Light
                searchWindow = 11;
                patchSize = 3;
                h = 10.0f;
                break;
            case 2: // Medium
                searchWindow = 15;
                patchSize = 5;
                h = 15.0f;
                break;
            case 3: // Strong
                searchWindow = 21;
                patchSize = 7;
                h = 20.0f;
                break;
            default:
                searchWindow = 15;
                patchSize = 5;
                h = 15.0f;
        }
        
        nlmDenoiseKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height,
                                                channels, searchWindow, patchSize, h);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    unsigned char* h_output = (unsigned char*)malloc(imageSize);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    // Determine output format based on input extension
    const char* ext = strrchr(inputPath, '.');
    if (ext && (strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0)) {
        stbi_write_png(outputPath, width, height, channels, h_output, width * channels);
    } else {
        stbi_write_jpg(outputPath, width, height, channels, h_output, 95);
    }
    
    free(h_output);
    stbi_image_free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

bool isImageFile(const char* filename) {
    const char* ext = strrchr(filename, '.');
    if (!ext) return false;
    
    return (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".JPG") == 0 ||
            strcmp(ext, ".jpeg") == 0 || strcmp(ext, ".JPEG") == 0 ||
            strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <input_directory> <output_directory>\n", argv[0]);
        return 1;
    }
    
    const char* inputDir = argv[1];
    const char* outputDir = argv[2];
    
    // Create output directory
    mkdir(outputDir, 0755);
    
    // Choose filter
    int filterType;
    printf("Choose filter:\n");
    printf("  1. Gaussian Blur\n");
    printf("  2. Sharpen\n");
    printf("  3. Denoise (Bilateral - Fast)\n");
    printf("  4. Denoise (Non-Local Means - Quality)\n");
    printf("Enter choice (1-4): ");
    scanf("%d", &filterType);
    
    if (filterType < 1 || filterType > 4) {
        printf("Invalid choice!\n");
        return 1;
    }
    
    int intensity = 2; // Default medium
    
    if (filterType == 1) {
        printf("\nChoose blur intensity:\n");
        printf("  1. Light blur (5x5 kernel, sigma=1.0)\n");
        printf("  2. Medium blur (7x7 kernel, sigma=2.0)\n");
        printf("  3. Strong blur (9x9 kernel, sigma=3.0)\n");
        printf("  4. Very strong blur (11x11 kernel, sigma=4.0)\n");
        printf("  5. Extreme blur (15x15 kernel, sigma=5.0)\n");
        printf("Enter choice (1-5): ");
        scanf("%d", &intensity);
        
        if (intensity < 1 || intensity > 5) {
            printf("Invalid choice! Using medium blur.\n");
            intensity = 2;
        }
        initializeKernels(intensity);
    } else if (filterType == 3 || filterType == 4) {
        printf("\nChoose denoise strength:\n");
        printf("  1. Light (preserve more detail, less noise removal)\n");
        printf("  2. Medium (balanced)\n");
        printf("  3. Strong (more aggressive, smoother result)\n");
        printf("Enter choice (1-3): ");
        scanf("%d", &intensity);
        
        if (intensity < 1 || intensity > 3) {
            printf("Invalid choice! Using medium strength.\n");
            intensity = 2;
        }
        
        if (filterType == 3) {
            printf("  Using Bilateral filter (fast, edge-preserving)\n");
        } else {
            printf("  Using Non-Local Means (slower, higher quality)\n");
        }
    } else {
        initializeKernels(intensity);
    }
    
    const char* filterName[] = {"", "Gaussian Blur", "Sharpen", 
                                "Bilateral Denoise", "NLM Denoise"};
    printf("\nProcessing images with %s filter...\n\n", filterName[filterType]);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    DIR* dir = opendir(inputDir);
    if (!dir) {
        fprintf(stderr, "Cannot open directory: %s\n", inputDir);
        return 1;
    }
    
    int imageCount = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (!isImageFile(entry->d_name)) continue;
        
        char inputPath[1024], outputPath[1024];
        snprintf(inputPath, sizeof(inputPath), "%s/%s", inputDir, entry->d_name);
        snprintf(outputPath, sizeof(outputPath), "%s/%s", outputDir, entry->d_name);
        
        processImage(inputPath, outputPath, filterType, intensity);
        imageCount++;
    }
    
    closedir(dir);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    printf("\n=====================================\n");
    printf("Processing complete!\n");
    printf("Total images processed: %d\n", imageCount);
    printf("Total time: %.3f seconds\n", duration.count() / 1000.0);
    printf("Average time per image: %.3f ms\n", 
           imageCount > 0 ? (double)duration.count() / imageCount : 0.0);
    printf("Output directory: %s\n", outputDir);
    printf("=====================================\n");
    
    return 0;
}
