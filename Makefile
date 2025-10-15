NVCC := nvcc
NVCCFLAGS := -std=c++20 -O3 -arch=sm_80
INCLUDES := -I./include

SRCS := $(wildcard src/*.cc src/*.cu)
KERNELS := $(wildcard kernels/*.cu)

TARGET := gpu-memperf

all: $(TARGET)

$(TARGET): $(SRCS) $(KERNELS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SRCS) $(KERNELS) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
