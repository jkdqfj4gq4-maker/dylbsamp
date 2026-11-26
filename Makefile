ARCHS = arm64 arm64e
TARGET = iphone:clang:latest:14.0

include $(THEOS)/makefiles/common.mk

TWEAK_NAME = resa_ao
resa_ao_FILES = resa_ao.c fishhook.c
resa_ao_CFLAGS = -DGLES_SILENCE_DEPRECATION -Wno-deprecated-declarations

include $(THEOS_MAKE_PATH)/tweak.mk
