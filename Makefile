ARCHS = arm64 arm64e
TARGET = iphone:clang:latest:14.0

include $(THEOS)/makefiles/common.mk

TWEAK_NAME = resa_probe
resa_probe_FILES = resa_probe.c fishhook.c
resa_probe_FRAMEWORKS = Foundation
resa_probe_CFLAGS = -fobjc-arc -DGLES_SILENCE_DEPRECATION -Wno-deprecated-declarations

include $(THEOS_MAKE_PATH)/tweak.mk
