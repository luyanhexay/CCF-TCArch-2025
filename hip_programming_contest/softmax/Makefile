HIPCC = hipcc

TARGET = softmax

SRCS = main.cpp kernel.hip

HEADERS = main.h

# 优化与架构设置（可通过环境变量覆盖）
HIPARCH ?= gfx908
CXXFLAGS ?= -O3 -ffast-math -fno-math-errno -fno-trapping-math -DNDEBUG
HIPFLAGS ?= --offload-arch=$(HIPARCH)

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o
