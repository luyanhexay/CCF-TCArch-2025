HIPCC = hipcc
CXX = g++

TARGET = apsp

HIP_SRCS = kernel.hip blocked_floyd_warshall_hip.hip apsp_solver_kernels.hip
CPP_SRCS = main.cpp apsp_solver.cpp

HEADERS = main.h apsp_solver.h

CXXFLAGS = -O2
HIPFLAGS = -O2

all: $(TARGET)

$(TARGET): $(HIP_SRCS) $(CPP_SRCS) $(HEADERS)
	$(HIPCC) $(HIPFLAGS) $(HIP_SRCS) $(CPP_SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o