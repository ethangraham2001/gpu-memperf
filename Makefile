NVCC := nvcc
CXX := g++

CXXFLAGS := -std=c++20 -O3
NVCCFLAGS := $(CXXFLAGS)
NVCCFLAGS += -Werror all-warnings
NVCCFLAGS += -lnvidia-ml
NVCCFLAGS += -Xcompiler "-Wall,-Wextra,-Wconversion,-Wshadow"
NVCCFLAGS += -Xcompiler "-Wno-unused-parameter"  # Often unavoidable in CUDA
INCLUDES := -I./include

# # Target multiple architectures for portability
# GENCODES := \
#   -gencode arch=compute_70,code=sm_70 \
#   -gencode arch=compute_75,code=sm_75 \
#   -gencode arch=compute_80,code=sm_80 \
#   -gencode arch=compute_89,code=sm_89

SRCS := $(wildcard src/*.cc src/*.cu)
TESTS := $(wildcard test/*.cpp)
KERNELS := $(wildcard kernels/*.cu)
HEADERS := $(wildcard include/*.hh include/*.cuh)

TARGET := gpu-memperf
TEST_TARGET := test-memperf

all: $(TARGET)

$(TARGET): $(SRCS) $(KERNELS) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SRCS) $(KERNELS) -o $(TARGET)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TESTS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TESTS) -o $(TEST_TARGET)

clean:
	rm -f $(TARGET) $(TEST_TARGET)

.PHONY: all clean
