# Copyright (c) 2015-2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly prohibited.

NV_TOPDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))../..)
NV_PLATFORM_DIR = $(NV_TOPDIR)/drive-linux

# ape directory, includes

NV_PLATFORM_APE_TOP_DIR = $(NV_PLATFORM_DIR)/samples/ape
NV_PLATFORM_APE_NVFX_INC = $(NV_PLATFORM_APE_TOP_DIR)/libs/include
NV_PLATFORM_APE_COMMON_INC = $(NV_PLATFORM_APE_TOP_DIR)/include
NV_PLATFORM_APE_LK_INC = $(NV_PLATFORM_APE_COMMON_INC)/lk
NV_PLATFORM_APE_TARGET_LIBS = $(NV_PLATFORM_APE_TOP_DIR)/make/target/libs
NV_PLATFORM_APE_LOADER_DIR = $(NV_PLATFORM_APE_TOP_DIR)/make/target/loader
# ape flags

NV_PLATFORM_APE_CFLAGS = -Os \
                         -fno-builtin \
                         -finline \
                         -ffunction-sections \
                         -fdata-sections \
                         -fno-short-enums \
                         -fno-common \
                         -fno-optimize-sibling-calls

NV_PLATFORM_APE_CFLAGS += -W -Wall -Werror \
                          -Wno-multichar \
                          -Wno-unused-parameter \
                          -Wno-unused-function \
                          -Wstrict-prototypes \
                          -Werror-implicit-function-declaration

NV_PLATFORM_APE_CFLAGS += -mcpu=cortex-a9 \
                          -mfpu=neon -mfloat-abi=softfp \
                          -mthumb-interwork \
                          -mthumb \
                          -mno-unaligned-access \
                          --std=gnu99

NV_PLATFORM_APE_CFLAGS += -DNVFX_FRAMEWORK=1 \
                          -DNVFX_ADSP_OFFLOAD=1 \
                          -D__thumb__

# ape build utilities

NV_PLATFORM_APE_LINKER_SCRIPT = $(NV_PLATFORM_APE_TOP_DIR)/ape-app-segments.ld
NV_PLATFORM_APE_STACK_USAGE = $(NV_PLATFORM_APE_TOP_DIR)/stackusage

# compiler utilities

LIBGCC = $(NV_TOPDIR)/toolchains/armv5-eabi--glibc--stable-2020.08-1/lib/gcc/arm-buildroot-linux-gnueabi/9.3.0/libgcc.a
LIBSTDC = $(NV_TOPDIR)/toolchains/armv5-eabi--glibc--stable-2020.08-1/arm-buildroot-linux-gnueabi/sysroot/usr/lib/libstdc++.a

CROSSBIN = $(NV_TOPDIR)/toolchains/armv5-eabi--glibc--stable-2020.08-1/bin/arm-buildroot-linux-gnueabi-
CC := $(CCACHE) $(CROSSBIN)gcc
CXX := $(CCACHE) $(CROSSBIN)g++
LD := $(CROSSBIN)ld
AR := $(CROSSBIN)ar
OBJDUMP := $(CROSSBIN)objdump
OBJCOPY := $(CROSSBIN)objcopy
CPPFILT := $(CROSSBIN)c++filt
SIZE := $(CROSSBIN)size
NM := $(CROSSBIN)nm
