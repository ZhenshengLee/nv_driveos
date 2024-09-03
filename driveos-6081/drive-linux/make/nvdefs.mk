# Copyright (c) 2012-2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly prohibited.

NV_TOPDIR                 := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))../..)
NV_PLATFORM_DIR            = $(NV_TOPDIR)/drive-linux
NV_PLATFORM_SAFETY         = 0
NV_PLATFORM_CUDA_TOOLKIT   ?= /usr/local/cuda-11.4

NV_KERNDIR         = $(NV_PLATFORM_DIR)/kernel

NV_PLATFORM_OS     = Linux

NV_HOST_OS = $(shell uname)
ifeq ($(NV_HOST_OS),Linux)
   # Linux environment
   NV_HOST_OSTYPE = linux
else
   # Windows environment
   NV_HOST_OSTYPE = win32
endif


NV_IS_DEBUG ?= 0

#Determine if debug or release (default)
ifeq ($(NV_IS_DEBUG),0)
NV_PLATFORM_OPT    = -Os
NV_PLATFORM_CFLAGS = -O2 \
                  -fomit-frame-pointer \
                  -finline-functions \
                  -finline-limit=300 \
                  -fgcse-after-reload
else
NV_PLATFORM_OPT    =
NV_PLATFORM_CFLAGS = -g
endif

#Append common cflags
NV_PLATFORM_CFLAGS += -fno-strict-aliasing \
                -Wall \
                -mharden-sls=all \
                -Wcast-align

NV_PLATFORM_CPPFLAGS  = -DWIN_INTERFACE_CUSTOM

NV_PLATFORM_LDFLAGS     = -Wl,--dynamic-linker=/lib/ld-linux-aarch64.so.1

NV_PLATFORM_SDK_INC_DIR = $(NV_PLATFORM_DIR)/include
NV_PLATFORM_SDK_LIB_DIR = $(NV_PLATFORM_DIR)/lib-target

NV_PLATFORM_SDK_INC   = -I$(NV_PLATFORM_SDK_INC_DIR) \
                        -I$(NV_PLATFORM_CUDA_TOOLKIT)/targets/aarch64-linux/include
NV_PLATFORM_SDK_LIB   = -L$(NV_PLATFORM_SDK_LIB_DIR) \
                        -L$(NV_PLATFORM_SDK_LIB_DIR)/$(NV_WINSYS) \
                        -Wl,-rpath-link=$(NV_PLATFORM_SDK_LIB_DIR) \
                        -Wl,-rpath-link=$(NV_PLATFORM_SDK_LIB_DIR)/$(NV_WINSYS)

NV_PLATFORM_MATHLIB   = -lm
NV_PLATFORM_THREADLIB = -lpthread

# separately installed open source packages

NV_PLATFORM_FFI_DIR       = $(NV_PLATFORM_DIR)/oss/ffi/libffi
NV_PLATFORM_ASOUND_DIR    = $(NV_PLATFORM_DIR)/oss/asound/libasound2
NV_PLATFORM_GLIB_DIR      = $(NV_PLATFORM_DIR)/oss/glib/libglib2.0
NV_PLATFORM_LIBXML2_DIR   = $(NV_PLATFORM_DIR)/oss/xml2/libxml2
NV_PLATFORM_LIBLZMA_DIR   = $(NV_PLATFORM_DIR)/oss/liblzma/liblzma
NV_PLATFORM_PCRE_DIR      = $(NV_PLATFORM_DIR)/oss/pcre/libpcre3
NV_PLATFORM_ZLIB_DIR      = $(NV_PLATFORM_DIR)/oss/zlib/zlib1g

# determine what ARCH flavor is being built against
ARM_ARCH_DIST = aarch64-linux-gnu

# compiler utilities

CROSSBIN = $(NV_TOPDIR)/toolchains/aarch64--glibc--stable-2022.03-1//bin/aarch64-buildroot-linux-gnu-

# override CROSSBIN in vars.mk
# need shell to expand quotes in NV_PLATFORM_DIR
-include $(shell echo $(NV_PLATFORM_DIR)/make/vars.mk)

# CC, CXX, AR, LD are implicitly set to hosts tools by gnu make
# https://ftp.gnu.org/old-gnu/Manuals/make-3.79.1/html_chapter/make_10.html#SEC96
# we check the same to force assign cross-toolchain equivalent
# For yocto builds, they are already set to cross-toolchain
# so, ignore force setting in such cases
#CC
ifeq ($(CC),cc)
CC     = $(CROSSBIN)gcc
endif

#CXX
ifeq ($(CXX),g++)
CXX    = $(CROSSBIN)g++
endif

#AR
ifeq ($(AR),ar)
AR     = $(CROSSBIN)ar
endif

#LD
ifeq ($(LD),ld)
LD = $(if $(wildcard *.cpp),$(CXX),$(CC))
endif

#RANLIB, STRIP, NM are empty by default
RANLIB ?= $(CROSSBIN)ranlib
STRIP  ?= $(CROSSBIN)strip
NM     ?= $(CROSSBIN)nm

# By default we assume non-yocto builds (i.e. make outside yocto)
IS_YOCTO_ENV := 0

ifneq ($(shell [ "$(LD)" = "$(CROSSBIN)gcc" -o "$(LD)" = "$(CROSSBIN)g++" ] && [ "$(CC)" = "$(CROSSBIN)gcc" ] && [ "$(CXX)" = "$(CROSSBIN)g++" ] &&  [ "$(AR)" = "$(CROSSBIN)ar" ] && [ "$(RANLIB)" = "$(CROSSBIN)ranlib" ] && [ "$(STRIP)" = "$(CROSSBIN)strip" ] && [ "$(NM)" = "$(CROSSBIN)nm" ] && echo true), true)
    $(warning Current environment seems different preset values in nvdefs.mk)
    $(warning using CC      = $(CC))
    $(warning using CXX     = $(CXX))
    $(warning using AR      = $(AR))
    $(warning using LD      = $(LD))
    $(warning using RANLIB  = $(RANLIB))
    $(warning using STRIP   = $(STRIP))
    $(warning using NM      = $(NM))
    $(warning If this is not intended please unset and re-make)

	# If CC, LD, etc are not in defaults, then we are in yocto ENV
	IS_YOCTO_ENV := 1
endif

# Only define these if it is not-yocto builds
ifneq ($(IS_YOCTO_ENV),1)
NV_PLATFORM_TARGET_INC_DIRS = \
        $(NV_PLATFORM_DIR)/targetfs/usr/include
NV_PLATFORM_TARGET_LIB_DIRS = \
        $(NV_PLATFORM_DIR)/targetfs/lib \
        $(NV_PLATFORM_DIR)/targetfs/usr/lib \
        $(NV_PLATFORM_DIR)/targetfs/usr/lib/tls

NV_PLATFORM_TARGET_INC = $(addprefix -I,$(NV_PLATFORM_TARGET_INC_DIRS))
NV_PLATFORM_TARGET_LIB = $(addprefix -L,$(NV_PLATFORM_TARGET_LIB_DIRS)) \
                         $(addprefix -Xlinker -rpath-link -Xlinker , \
                                     $(NV_PLATFORM_TARGET_LIB_DIRS))
endif

STRINGIFY = /bin/sed -e 's/"/\\"/g' -e 's|^.*$$|"&\\n"|'

%.glslvh: %.glslv
	/bin/cat $(filter %.h,$^) $(filter %.glslv,$^) | \
          $(STRINGIFY) > $@

%.glslfh: %.glslf
	/bin/cat $(filter %.h,$^) $(filter %.glslf,$^) | \
          $(STRINGIFY) > $@

# support for windowing system subdirs

NV_LIST_WINSYS := egldevice wayland x11 direct-to-display
ifndef NV_WINSYS
  NV_WINSYS := x11
  ifneq ($(NV_WINSYS),$(NV_LIST_WINSYS))
    $(warning Defaulting NV_WINSYS to x11; legal values are: $(NV_LIST_WINSYS))
  endif
endif

ifeq ($(NV_WINSYS),egldevice)
   NV_PLATFORM_CPPFLAGS +=
   NV_PLATFORM_WINSYS_LIBS = -ldl
else ifeq ($(NV_WINSYS),direct-to-display)
   NV_PLATFORM_CPPFLAGS += -DVK_USE_PLATFORM_DISPLAY_KHR
else ifeq ($(NV_WINSYS),wayland)
   NV_PLATFORM_CPPFLAGS += -DWAYLAND
   NV_PLATFORM_CPPFLAGS  += -DENABLE_IVI_SHELL
   NV_PLATFORM_CPPFLAGS  += -DVK_USE_PLATFORM_WAYLAND_KHR
   NV_PLATFORM_WINSYS_LIBS = \
        -lxkbcommon -lwayland-client -lwayland-egl \
        -lilmClient \
        -Wl,-rpath-link=$(NV_PLATFORM_LIBLZMA_DIR)/lib/aarch64-linux-gnu:$(NV_PLATFORM_ZLIB_DIR)/lib/aarch64-linux-gnu:$(NV_PLATFORM_LIBXML2_DIR)/usr/lib/aarch64-linux-gnu
else ifeq ($(NV_WINSYS),x11)
   NV_PLATFORM_CPPFLAGS += -DX11
   NV_PLATFORM_WINSYS_LIBS = -lX11 -lXau
   NV_PLATFORM_CPPFLAGS  += -DVK_USE_PLATFORM_XLIB_KHR
else
   $(error Invalid NV_WINSYS value: $(NV_WINSYS))
endif

$(NV_WINSYS)/%.o : %.c
	@mkdir -p $(NV_WINSYS)
	$(COMPILE.c) $(OUTPUT_OPTION) $<

$(NV_WINSYS)/%.o : %.cpp
	@mkdir -p $(NV_WINSYS)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<
