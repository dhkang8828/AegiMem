# Makefile for Memory Agent C Library

CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -I../../include
LDFLAGS = -shared

# Targets
TARGET = libmemory_agent.so
TEST_TARGET = test_memory_agent

# Source files
SOURCES = memory_agent.c
OBJECTS = $(SOURCES:.c=.o)

# Test source
TEST_SOURCES = test_memory_agent.c
TEST_OBJECTS = $(TEST_SOURCES:.c=.o)

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^
	@echo "Built shared library: $(TARGET)"

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJECTS) $(TARGET)
	$(CC) -o $@ $(TEST_OBJECTS) -L. -lmemory_agent -Wl,-rpath,.
	@echo "Built test program: $(TEST_TARGET)"

clean:
	rm -f $(OBJECTS) $(TEST_OBJECTS) $(TARGET) $(TEST_TARGET)
	@echo "Cleaned build artifacts"

install: $(TARGET)
	@echo "Installing $(TARGET) to /usr/local/lib..."
	sudo cp $(TARGET) /usr/local/lib/
	sudo ldconfig
	@echo "Installation complete"

.PHONY: help
help:
	@echo "Memory Agent C Library Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build shared library (default)"
	@echo "  test      - Build test program"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install library to system"
	@echo "  help      - Show this help message"
