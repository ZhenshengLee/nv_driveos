# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include ../../../make/nvdefs.mk

TARGETS = nvm_ijpe_sci

CFLAGS   := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS) -I. -I../../../include/nvmedia_6x/ -I../utils
CPPFLAGS := $(NV_PLATFORM_SDK_INC) $(NV_PLATFORM_CPPFLAGS)
LDFLAGS  := $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_TARGET_LIB) $(NV_PLATFORM_LDFLAGS)

OBJS   := image_jpegenc.o
OBJS   += cmdline.o
OBJS   += ../utils/scibuf_utils.o
OBJS   += ../utils/config_parser.o
OBJS   += ../utils/log_utils.o
OBJS   += ../utils/misc_utils.o

LDLIBS := -lnvscibuf
LDLIBS += -lnvscisync
LDLIBS += -lnvmedia_ijpe_sci
LDLIBS += -lz
LDLIBS += -lm

CFLAGS  += -D_FILE_OFFSET_BITS=64

$(TARGETS): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean clobber:
	rm -rf $(OBJS) $(TARGETS)
