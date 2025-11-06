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

// Sobel kernels for edge detection
__constant__ float d_sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ float d_sobelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

// CUDA kernel for Gaussian blur
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

// Bilateral filter kernel
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
                
                float spatialDist = kx * kx + ky * ky;
                float spatialWeight = expf(spatialDist * spatialCoeff);
                
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
        
        for (int sy = -searchRadius; sy <= searchRadius; sy++) {
            for (int sx = -searchRadius; sx <= searchRadius; sx++) {
                int searchX = min(max(x + sx, 0), width - 1);
                int searchY = min(max(y + sy, 0), height - 1);
                
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

// Unsharp mask for deblurring
__global__ void unsharpMaskKernel(unsigned char* input, unsigned char* output,
                                  int width, int height, int channels,
                                  float amount, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int radius = 1;
    
    for (int c = 0; c < channels; c++) {
        float blurred = 0.0f;
        float weightSum = 0.0f;
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                
                int idx = (iy * width + ix) * channels + c;
                float weight = 1.0f / 9.0f;
                
                blurred += input[idx] * weight;
                weightSum += weight;
            }
        }
        
        blurred /= weightSum;
        
        int centerIdx = (y * width + x) * channels + c;
        float original = (float)input[centerIdx];
        float detail = original - blurred;
        
        if (fabsf(detail) < threshold) {
            detail = 0.0f;
        }
        
        float sharpened = original + amount * detail;
        output[centerIdx] = (unsigned char)min(max(sharpened, 0.0f), 255.0f);
    }
}

// Sobel edge detection
__global__ void sobelEdgeKernel(unsigned char* input, unsigned char* output,
                                int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float gray = 0.0f;
    int baseIdx = (y * width + x) * channels;
    
    if (channels >= 3) {
        gray = 0.299f * input[baseIdx] + 
               0.587f * input[baseIdx + 1] + 
               0.114f * input[baseIdx + 2];
    } else {
        gray = input[baseIdx];
    }
    
    float gx = 0.0f, gy = 0.0f;
    
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);
            
            int idx = (iy * width + ix) * channels;
            float pixel;
            
            if (channels >= 3) {
                pixel = 0.299f * input[idx] + 
                       0.587f * input[idx + 1] + 
                       0.114f * input[idx + 2];
            } else {
                pixel = input[idx];
            }
            
            int kernelIdx = (ky + 1) * 3 + (kx + 1);
            gx += pixel * d_sobelX[kernelIdx];
            gy += pixel * d_sobelY[kernelIdx];
        }
    }
    
    float magnitude = sqrtf(gx * gx + gy * gy);
    magnitude = fminf(magnitude, 255.0f);
    
    for (int c = 0; c < channels; c++) {
        output[(y * width + x) * channels + c] = (unsigned char)magnitude;
    }
}

// Black and white conversion
__global__ void blackWhiteKernel(unsigned char* input, unsigned char* output,
                                 int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * channels;
    
    if (channels >= 3) {
        float gray = 0.299f * input[idx] + 
                    0.587f * input[idx + 1] + 
                    0.114f * input[idx + 2];
        
        unsigned char grayValue = (unsigned char)gray;
        
        for (int c = 0; c < channels; c++) {
            output[idx + c] = grayValue;
        }
    } else {
        output[idx] = input[idx];
    }
}

// Auto color balance kernel
__global__ void autoColorBalanceKernel(unsigned char* input, unsigned char* output,
                                       int width, int height, int channels,
                                       float* minVals, float* maxVals) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * channels;
    
    for (int c = 0; c < channels; c++) {
        if (c >= 3) {
            output[idx + c] = input[idx + c];
            continue;
        }
        
        float value = (float)input[idx + c];
        float minVal = minVals[c];
        float maxVal = maxVals[c];
        
        float normalized = 0.0f;
        if (maxVal > minVal) {
            normalized = 255.0f * (value - minVal) / (maxVal - minVal);
        } else {
            normalized = value;
        }
        
        output[idx + c] = (unsigned char)min(max(normalized, 0.0f), 255.0f);
    }
}

void computeColorStats(unsigned char* h_image, int width, int height, int channels,
                       float* minVals, float* maxVals) {
    const int histSize = 256;
    int hist[3][histSize] = {0};
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            for (int c = 0; c < 3 && c < channels; c++) {
                hist[c][h_image[idx + c]]++;
            }
        }
    }
    
    int totalPixels = width * height;
    int lowThreshold = totalPixels * 0.01f;
    int highThreshold = totalPixels * 0.99f;
    
    for (int c = 0; c < 3 && c < channels; c++) {
        int cumSum = 0;
        minVals[c] = 0;
        maxVals[c] = 255;
        
        for (int i = 0; i < histSize; i++) {
            cumSum += hist[c][i];
            if (cumSum >= lowThreshold) {
                minVals[c] = i;
                break;
            }
        }
        
        cumSum = 0;
        for (int i = histSize - 1; i >= 0; i--) {
            cumSum += hist[c][i];
            if (cumSum >= (totalPixels - highThreshold)) {
                maxVals[c] = i;
                break;
            }
        }
    }
}

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
    
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

void initializeKernels(int blurIntensity) {
    int kernelSize;
    float sigma;
    
    switch(blurIntensity) {
        case 1: kernelSize = 5; sigma = 1.0f; break;
        case 2: kernelSize = 7; sigma = 2.0f; break;
        case 3: kernelSize = 9; sigma = 3.0f; break;
        case 4: kernelSize = 11; sigma = 4.0f; break;
        case 5: kernelSize = 15; sigma = 5.0f; break;
        default: kernelSize = 7; sigma = 2.0f;
    }
    
    printf("  Blur settings: Kernel size = %dx%d, Sigma = %.1f\n", 
           kernelSize, kernelSize, sigma);
    
    float* h_gaussianKernel = (float*)malloc(kernelSize * kernelSize * sizeof(float));
    generateGaussianKernel(h_gaussianKernel, kernelSize, sigma);
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_gaussianKernel, h_gaussianKernel, 
                                   kernelSize * kernelSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernelSize, &kernelSize, sizeof(int)));
    
    free(h_gaussianKernel);
    
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
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                  (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    if (filterType == 1) {
        int kernelSize;
        CUDA_CHECK(cudaMemcpyFromSymbol(&kernelSize, d_kernelSize, sizeof(int)));
        gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, 
                                                     channels, kernelSize);
    } else if (filterType == 2) {
        sharpenKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    } else if (filterType == 3) {
        int windowSize;
        float sigmaSpatial, sigmaRange;
        
        switch(intensity) {
            case 1: windowSize = 5; sigmaSpatial = 3.0f; sigmaRange = 50.0f; break;
            case 2: windowSize = 7; sigmaSpatial = 5.0f; sigmaRange = 75.0f; break;
            case 3: windowSize = 9; sigmaSpatial = 7.0f; sigmaRange = 100.0f; break;
            default: windowSize = 7; sigmaSpatial = 5.0f; sigmaRange = 75.0f;
        }
        
        bilateralFilterKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height,
                                                        channels, windowSize, 
                                                        sigmaSpatial, sigmaRange);
    } else if (filterType == 4) {
        int searchWindow, patchSize;
        float h;
        
        switch(intensity) {
            case 1: searchWindow = 11; patchSize = 3; h = 10.0f; break;
            case 2: searchWindow = 15; patchSize = 5; h = 15.0f; break;
            case 3: searchWindow = 21; patchSize = 7; h = 20.0f; break;
            default: searchWindow = 15; patchSize = 5; h = 15.0f;
        }
        
        nlmDenoiseKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height,
                                                   channels, searchWindow, patchSize, h);
    } else if (filterType == 5) {
        float amount, threshold;
        
        switch(intensity) {
            case 1: amount = 0.5f; threshold = 5.0f; break;
            case 2: amount = 1.0f; threshold = 3.0f; break;
            case 3: amount = 1.5f; threshold = 2.0f; break;
            default: amount = 1.0f; threshold = 3.0f;
        }
        
        unsharpMaskKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height,
                                                    channels, amount, threshold);
    } else if (filterType == 6) {
        sobelEdgeKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    } else if (filterType == 7) {
        blackWhiteKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    } else if (filterType == 8) {
        float minVals[3], maxVals[3];
        computeColorStats(h_input, width, height, channels, minVals, maxVals);
        
        float *d_minVals, *d_maxVals;
        CUDA_CHECK(cudaMalloc(&d_minVals, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_maxVals, 3 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_minVals, minVals, 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_maxVals, maxVals, 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        autoColorBalanceKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height,
                                                         channels, d_minVals, d_maxVals);
        
        CUDA_CHECK(cudaFree(d_minVals));
        CUDA_CHECK(cudaFree(d_maxVals));
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    unsigned char* h_output = (unsigned char*)malloc(imageSize);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));
    
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
    
    mkdir(outputDir, 0755);
    
    int filterType;
    printf("╔═══════════════════════════════════════════════════╗\n");
    printf("║     CUDA Image Manipulation Toolkit               ║\n");
    printf("╚═══════════════════════════════════════════════════╝\n\n");
    printf("Choose filter:\n");
    printf("  1. Gaussian Blur\n");
    printf("  2. Sharpen\n");
    printf("  3. Denoise (Bilateral - Fast)\n");
    printf("  4. Denoise (Non-Local Means - Quality)\n");
    printf("  5. Deblur (Unsharp Mask)\n");
    printf("  6. Edge Detection (Sobel)\n");
    printf("  7. Black & White\n");
    printf("  8. Auto Color Balance\n");
    printf("\nEnter choice (1-8): ");
    scanf("%d", &filterType);
    
    if (filterType < 1 || filterType > 8) {
        printf("Invalid choice!\n");
        return 1;
    }
    
    int intensity = 2;
    
    if (filterType == 1) {
        printf("\nChoose blur intensity:\n");
        printf("  1. Light (5x5, sigma=1.0)\n");
        printf("  2. Medium (7x7, sigma=2.0)\n");
        printf("  3. Strong (9x9, sigma=3.0)\n");
        printf("  4. Very Strong (11x11, sigma=4.0)\n");
        printf("  5. Extreme (15x15, sigma=5.0)\n");
        printf("Enter choice (1-5): ");
        scanf("%d", &intensity);
        if (intensity < 1 || intensity > 5) intensity = 2;
        initializeKernels(intensity);
    } else if (filterType == 3 || filterType == 4) {
        printf("\nChoose denoise strength:\n");
        printf("  1. Light\n");
        printf("  2. Medium\n");
        printf("  3. Strong\n");
        printf("Enter choice (1-3): ");
        scanf("%d", &intensity);
        if (intensity < 1 || intensity > 3) intensity = 2;
    } else if (filterType == 5) {
        printf("\nChoose deblur strength:\n");
        printf("  1. Subtle (amount=0.5)\n");
        printf("  2. Medium (amount=1.0)\n");
        printf("  3. Strong (amount=1.5)\n");
        printf("Enter choice (1-3): ");
        scanf("%d", &intensity);
        if (intensity < 1 || intensity > 3) intensity = 2;
    } else {
        initializeKernels(intensity);
    }
    
    const char* filterName[] = {"", "Gaussian Blur", "Sharpen", 
                                "Bilateral Denoise", "NLM Denoise",
                                "Deblur", "Edge Detection", 
                                "Black & White", "Auto Color Balance"};
    printf("\nProcessing images with %s...\n\n", filterName[filterType]);
    
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
    
    printf("\n╔═══════════════════════════════════════════════════╗\n");
    printf("║           Processing Complete!                    ║\n");
    printf("╚═══════════════════════════════════════════════════╝\n");
    printf("Total images processed: %d\n", imageCount);
    printf("Total time: %.3f seconds\n", duration.count() / 1000.0);
    printf("Average time per image: %.3f ms\n", 
           imageCount > 0 ? (double)duration.count() / imageCount : 0.0);
    printf("Output directory: %s\n", outputDir);
    printf("═══════════════════════════════════════════════════\n");
    
    return 0;
}
