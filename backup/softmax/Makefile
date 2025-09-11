HIPCC = hipcc

TARGET = softmax

SRCS = main.cpp kernel.hip

HEADERS = main.h

# 优化与架构设置（可通过环境变量覆盖）
HIPARCH ?= gfx908
CXXFLAGS ?= -O3 -ffast-math -fno-math-errno -fno-trapping-math -DNDEBUG
HIPFLAGS ?= --offload-arch=$(HIPARCH)

# 支持参数化编译 - 使用优化后的默认值
TPB ?= 256
GRID_MULTIPLIER ?= 4
PINNED_MEMORY_THRESHOLD ?= 500000
CXXFLAGS += -DTPB=$(TPB) -DGRID_MULTIPLIER=$(GRID_MULTIPLIER) -DPINNED_MEMORY_THRESHOLD=$(PINNED_MEMORY_THRESHOLD)

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o
