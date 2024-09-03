# Copyright (c) 2017-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include ../../../make/nvdefs.mk

TARGETS = nvmimg_ldc

CFLAGS   := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS) -I. -I../utils
CFLAGS   += -I$(NV_PLATFORM_DIR)/include/nvmedia_6x
CPPFLAGS := $(NV_PLATFORM_SDK_INC) $(NV_PLATFORM_CPPFLAGS)
LDFLAGS  := $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_LDFLAGS)

OBJS   := imageldc.o
OBJS   += cmdline.o
OBJS   += ../utils/config_parser.o
OBJS   += ../utils/log_utils.o

LDLIBS := -lnvmedialdc
LDLIBS += -lnvscibuf
LDLIBS += -lnvscisync
LDLIBS += -lnvscicommon
LDLIBS += -lnvvic

$(TARGETS): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean clobber:
	rm -rf $(OBJS) $(TARGETS)
