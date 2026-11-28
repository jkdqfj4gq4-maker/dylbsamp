ARCHS = arm64 arm64e
TARGET = iphone:clang:latest:14.0

include $(THEOS)/makefiles/common.mk

TWEAK_NAME = resa_ao

# 1. Add your C files, the C++ bridge, and only the specific ReShade compiler files needed for GLSL
resa_ao_FILES = resa_ao.c fishhook.c resa_fx_bridge.cpp
resa_ao_FILES += reshadefx/src/effect_lexer.cpp \
                 reshadefx/src/effect_expression.cpp \
                 reshadefx/src/effect_parser_exp.cpp \
                 reshadefx/src/effect_parser_stmt.cpp \
                 reshadefx/src/effect_preprocessor.cpp \
                 reshadefx/src/effect_symbol_table.cpp \
                 reshadefx/src/effect_symbol_table_intrinsics.cpp \
                 reshadefx/src/effect_codegen_glsl.cpp

# 2. C Flags (for resa_ao.c and fishhook.c)
resa_ao_CFLAGS = -DGLES_SILENCE_DEPRECATION -Wno-deprecated-declarations

# 3. C++ Flags (for resa_fx_bridge.cpp and reshadefx)
# -Wno-error is CRITICAL here because DuckStation's code has strict warnings that Apple's compiler fails on
resa_ao_CCFLAGS = -std=c++17 -fexceptions -Wno-error -I./reshadefx/include

# 4. Link the C++ standard library
resa_ao_LIBRARIES = c++

# 5. Frameworks
resa_ao_FRAMEWORKS = UIKit Foundation OpenGLES

include $(THEOS_MAKE_PATH)/tweak.mk
