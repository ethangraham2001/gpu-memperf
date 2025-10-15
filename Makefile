NVCC := nvcc
NVCCFLAGS := -std=c++20 -O3 -arch=sm_80
INCLUDES := -I./include

SRCS := $(wildcard src/*.cc src/*.cu)
KERNELS := $(wildcard kernels/*.cu)
HEADERS := $(wildcard include/*.hh include/*.cuh)

TARGET := gpu-memperf

all: $(TARGET)

$(TARGET): $(SRCS) $(KERNELS) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SRCS) $(KERNELS) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
