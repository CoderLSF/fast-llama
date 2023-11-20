# Compiler
CC := g++
CXXFLAGS := -std=c++20

# 判断处理器架构并设置相应的编译参数
ifeq ($(shell uname -m),x86_64)
    DEFINES := -D_GNU_SOURCE
ifneq ($(shell grep -c avx512f /proc/cpuinfo),0)
    ARCH_FLAGS := -march=native -mavx512f -mavx512bw -mavx512vl -mavx512dq
else ifneq ($(shell grep -c avx2 /proc/cpuinfo),0)
    ARCH_FLAGS := -mavx2
else ifneq ($(shell grep -c sse4_2 /proc/cpuinfo),0)
    ARCH_FLAGS := -march=native -msse4.2
    endif
    ifneq ($(shell grep -c '^physical id' /proc/cpuinfo | sort -u | wc -l),1)
    LIBS += -lnuma
    else
    DEFINES += -DDISABLE_NUMA
    endif
else ifeq ($(shell uname -m),aarch64)
    ARCH_FLAGS := -march=armv8-a+simd -mcpu=native -mfpu=neon
    DEFINES += -DDISABLE_NUMA
endif

CFLAGS := $(ARCH_FLAGS) $(DEFINES) -Wall

LIBS += -lpthread -lm

TARGET := ./main
SRCS := $(shell find ./src -name "*.cpp")
OBJS := $(SRCS:.cpp=.o)
DEPS := $(SRCS:.cpp=.d)

INCLUDES := $(shell find ./src -name "*.h" -o -name "*.hpp" | xargs dirname | sort -u | awk '{printf "-I%s ", $$0}')

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(CXXFLAGS) $(CFLAGS) $(LIBS)

%.o: %.cpp
	$(CC) -c -o $@ $< $(CXXFLAGS) $(CFLAGS) $(INCLUDES)

clean:
	rm -f $(OBJS) $(TARGET)
