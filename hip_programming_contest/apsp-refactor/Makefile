HIPCC = hipcc

TARGET = apsp

SRCS = main.cpp solve.cpp blocked_floyd_warshall_cuda.hip

HEADERS = main.h

CXXFLAGS = -O2

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(HIPCC) $(CXXFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET) *.o