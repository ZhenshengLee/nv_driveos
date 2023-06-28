# Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include ../../../make/nvdefs.mk

TARGETS = nvmimg_2d

UTILS_DIR := $(NV_PLATFORM_DIR)/samples/nvmedia_6x/utils

CFLAGS := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS)
CFLAGS += -I$(UTILS_DIR) -I$(NV_PLATFORM_DIR)/include/nvmedia_6x
CFLAGS += -I$(NV_PLATFORM_DIR)/include
LDFLAGS := $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_LDFLAGS)

OBJS := \
	image_2d.o \
	$(UTILS_DIR)/config_parser.o \
	$(UTILS_DIR)/log_utils.o

LDLIBS := \
	-lnvmedia2d \
	-lnvscibuf \
	-lnvscisync

$(TARGETS): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean clobber:
	rm -rf $(OBJS) $(TARGETS)
