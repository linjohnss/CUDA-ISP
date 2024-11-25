# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -std=c++11 -O3

# Executable name
EXEC = cuda-isp

# Source files
SRC = main.cu

# Include directories
INCLUDES = -I./filters -I./stb_image

# Object files
OBJ = $(SRC:.cu=.o)

# Target to build the executable
all: $(EXEC)

$(EXEC): $(SRC)
	$(NVCC) $(INCLUDES) -o $@ $^

# Clean up build files
clean:
	rm -f $(EXEC) $(OBJ)
