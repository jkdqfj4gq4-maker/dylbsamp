ARCHS = arm64 arm64e
TARGET = iphone:clang:latest:14.0

include $(THEOS)/makefiles/common.mk

TWEAK_NAME = resa_ao_probe
resa_ao_probe_FILES = resa_ao_probe.m fishhook.c
resa_ao_probe_FRAMEWORKS = Foundation OpenGLES
resa_ao_probe_CFLAGS = -fobjc-arc -DGLES_SILENCE_DEPRECATION -Wno-deprecated-declarations

include $(THEOS_MAKE_PATH)/tweak.mk
