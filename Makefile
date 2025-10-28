NVCC := nvcc
NVCCFLAGS := -std=c++20 -O3
NVCCFLAGS += -Werror all-warnings
NVCCFLAGS += -Xcompiler "-Wall,-Wextra,-Wconversion,-Wshadow"
NVCCFLAGS += -Xcompiler "-Wno-unused-parameter"  # Often unavoidable in CUDA
INCLUDES := -I./include

# Target multiple architectures for portability
GENCODES := \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_89,code=sm_89

SRCS := $(wildcard src/*.cc src/*.cu)
KERNELS := $(wildcard kernels/*.cu)
HEADERS := $(wildcard include/*.hh include/*.cuh)

TARGET := gpu-memperf

all: $(TARGET)

$(TARGET): $(SRCS) $(KERNELS) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GENCODES) $(SRCS) $(KERNELS) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
