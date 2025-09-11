HIPCC = hipcc

TARGET = apsp

SRCS = main.cpp kernel.hip blocked_floyd_warshall_hip.hip

HEADERS = main.h

CXXFLAGS = -O2

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o