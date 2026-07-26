# ---- Config ----------------------------------------------------------
CXX      := g++
CC       := gcc
QDLDL_DIR := third_party/qdldl
QDLDL_INC := $(QDLDL_DIR)/include

CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -g -O0 -Iinclude -I$(QDLDL_INC)
# Third-party C: no strict warnings on code we don't own.
CFLAGS   := -g -O0 -I$(QDLDL_INC)

# ---- Layout ----------------------------------------------------------
SRC_DIR   := src
BUILD_DIR := build
TARGET    := $(BUILD_DIR)/ip_pmm

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

# Vendored QDLDL, compiled once as C.
QDLDL_OBJ := $(BUILD_DIR)/qdldl.o

# ---- Rules -----------------------------------------------------------
.PHONY: all clean run test

all: $(TARGET)

$(TARGET): $(OBJS) $(QDLDL_OBJ) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# QDLDL is C, so compile it with the C compiler (guarantees C linkage,
# which our extern "C" wrapper in kkt_solver.cpp relies on).
$(QDLDL_OBJ): $(QDLDL_DIR)/src/qdldl.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: all
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)


# ---- Tests -----------------------------------------------------------
TEST_DIR      := tests
TEST_TARGET   := $(BUILD_DIR)/run_tests

# Reuse the solver's object files, minus main.o (doctest brings its own main).
LIB_OBJS  := $(filter-out $(BUILD_DIR)/main.o,$(OBJS))
TEST_SRCS := $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJS := $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_DIR)/$(TEST_DIR)/%.o,$(TEST_SRCS))

$(BUILD_DIR)/$(TEST_DIR)/%.o: $(TEST_DIR)/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Ithird_party/doctest -MMD -MP -c $< -o $@

# The test binary also needs QDLDL (kkt_solver.o is in LIB_OBJS and calls it).
$(TEST_TARGET): $(LIB_OBJS) $(TEST_OBJS) $(QDLDL_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)


-include $(DEPS)