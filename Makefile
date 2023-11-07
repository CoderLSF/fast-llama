# Compiler
CC := g++
CXXFLAGS := -std=c++20

# Build flags
BUILD_FLAGS ?= -O3

# Directories
WORKDIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
SRCDIR := $(WORKDIR)/src
INCDIRS := $(shell find $(SRCDIR) -type d)

# Source files
SRCS := $(shell find $(SRCDIR) -name "*.cpp")
OBJS := $(SRCS:.cpp=.o)

# Includes
INCLUDES := $(addprefix -I,$(INCDIRS))

# Libraries
LIBS := -pthread -lm -lnuma

# Target
TARGET := $(WORKDIR)/main

# Compiler flags
CFLAGS := -mavx512f -mavx512bw -mavx512vl -mavx512dq -D_GNU_SOURCE -Wall $(INCLUDES)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CXXFLAGS) $(CFLAGS) $(BUILD_FLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CC) $(CXXFLAGS) $(CFLAGS) $(BUILD_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
