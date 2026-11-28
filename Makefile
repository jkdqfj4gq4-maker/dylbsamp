ARCHS = arm64 arm64e
TARGET = iphone:clang:latest:14.0

include $(THEOS)/makefiles/common.mk

TWEAK_NAME = resa_ao

# Grab ALL .cpp files in the reshadefx/src folder, but FILTER OUT the SPIR-V and HLSL 
# backends because we only need GLSL for iOS (and they require extra libraries).
RESHADE_SRC = $(filter-out reshadefx/src/effect_codegen_spirv.cpp reshadefx/src/effect_codegen_hlsl.cpp, $(wildcard reshadefx/src/*.cpp))

resa_ao_FILES = resa_ao.c fishhook.c resa_fx_bridge.cpp $(RESHADE_SRC)

# C Flags
resa_ao_CFLAGS = -DGLES_SILENCE_DEPRECATION -Wno-deprecated-declarations

# C++ Flags (-Wno-error is CRITICAL to bypass DuckStation's switch-case warnings)
resa_ao_CCFLAGS = -std=c++17 -fexceptions -Wno-error -I./reshadefx/include

# Link the C++ standard library
resa_ao_LIBRARIES = c++

# Frameworks
resa_ao_FRAMEWORKS = UIKit Foundation OpenGLES

include $(THEOS_MAKE_PATH)/tweak.mk
