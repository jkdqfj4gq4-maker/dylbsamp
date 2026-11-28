ARCHS = arm64 arm64e
TARGET = iphone:clang:latest:14.0

include $(THEOS)/makefiles/common.mk

TWEAK_NAME = resa_ao

# 1. Add your C files, the C++ bridge, and DuckStation's reshadefx source files
resa_ao_FILES = resa_ao.c fishhook.c resa_fx_bridge.cpp
resa_ao_FILES += $(wildcard reshadefx/src/*.cpp)

# 2. C Flags (for resa_ao.c and fishhook.c)
resa_ao_CFLAGS = -DGLES_SILENCE_DEPRECATION -Wno-deprecated-declarations

# 3. C++ Flags (for resa_fx_bridge.cpp and reshadefx)
resa_ao_CCFLAGS = -std=c++17 -fexceptions -I./reshadefx/include

# 4. Link the C++ standard library
resa_ao_LIBRARIES = c++

# 5. Frameworks
resa_ao_FRAMEWORKS = UIKit Foundation OpenGLES

include $(THEOS_MAKE_PATH)/tweak.mk
