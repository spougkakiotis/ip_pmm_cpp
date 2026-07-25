# ---- Config ----------------------------------------------------------
CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -g -O0 -Iinclude

# ---- Layout ----------------------------------------------------------
SRC_DIR   := src
BUILD_DIR := build
TARGET    := $(BUILD_DIR)/ip_pmm

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

# ---- Rules -----------------------------------------------------------
.PHONY: all clean run test

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

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

$(TEST_TARGET): $(LIB_OBJS) $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)


-include $(DEPS)