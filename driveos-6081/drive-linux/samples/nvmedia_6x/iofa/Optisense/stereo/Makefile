# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include ../../../../../make/nvdefs.mk

TARGETS = stereo_sense

CFLAGS   := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS)
CPPFLAGS := $(NV_PLATFORM_SDK_INC) $(NV_PLATFORM_CPPFLAGS) -I. -I../../../../../include/nvmedia_6x/ -I../common
LDFLAGS  := $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_TARGET_LIB) $(NV_PLATFORM_LDFLAGS)

CFLAGS  += -DNVMEDIA_NVSCI_ENABLE
CFLAGS  += -D_FILE_OFFSET_BITS=64
ifeq ($(NV_PLATFORM_OS), QNX)
        CFLAGS  += -DNVRM_SAFE_API=1
        CFLAGS  += -DNVMEDIA_QNX
endif

OBJS   := stereo_sense.o
OBJS   += stereo_cmdline.o
OBJS   += ../common/common_functions.o
OBJS   += ../common/image_reader.o
OBJS   += ../common/image_buffer.o
OBJS   += ../common/ofa_class.o
OBJS   += ../common/file_writer.o
OBJS   += ../common/median.o
OBJS   += ../common/upsample.o
OBJS   += ../common/lrcheck.o


LDLIBS := -lnvscibuf
LDLIBS += -lnvscisync
LDLIBS += -lnvmedia_iofa_sci

ifeq ($(NV_PLATFORM_OS), Linux)
    LDLIBS  += -lpthread
endif

$(TARGETS): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean clobber:
	rm -rf $(OBJS) $(TARGETS)

