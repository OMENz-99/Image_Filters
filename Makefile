# Makefile for CUDA Image Convolution Filter

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -std=c++11 -O3 -arch=sm_86
# Note: Adjust -arch=sm_75 based on your GPU architecture
# Common values: sm_60 (Pascal), sm_75 (Turing), sm_86 (Ampere), sm_89 (Ada Lovelace)

# Target executable
TARGET = image_filter

# Source files
SOURCES = main.cu

# Object files
OBJECTS = $(SOURCES:.cu=.o)

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Compile .cu files to object files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Run with example
run: $(TARGET)
	./$(TARGET) input_images output_images

# Check CUDA installation
check:
	@echo "Checking CUDA installation..."
	@nvcc --version
	@echo "\nChecking GPU devices..."
	@nvidia-smi

.PHONY: all clean run check
