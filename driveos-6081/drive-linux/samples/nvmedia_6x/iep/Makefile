# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include ../../../make/nvdefs.mk

TARGETS = nvm_iep_sci

CFLAGS   := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS) -I. -I../../../include/nvmedia_6x/ -I../utils
CPPFLAGS := $(NV_PLATFORM_SDK_INC) $(NV_PLATFORM_CPPFLAGS)
LDFLAGS  := $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_TARGET_LIB) $(NV_PLATFORM_LDFLAGS)

CPPFLAGS += -DNV_IS_SAFETY=$(NV_PLATFORM_SAFETY)
CFLAGS  += -DNVMEDIA_NVSCI_ENABLE
CFLAGS  += -D_FILE_OFFSET_BITS=64
ifeq ($(NV_PLATFORM_OS), QNX)
	CFLAGS  += -DNVRM_SAFE_API=1
	CFLAGS  += -DNVMEDIA_QNX
endif

OBJS   := image_encoder.o
OBJS   += cmdline.o
OBJS   += ../utils/scibuf_utils.o
OBJS   += ../utils/config_parser.o
OBJS   += ../utils/misc_utils.o
OBJS   += ../utils/thread_utils.o
OBJS   += ../utils/log_utils.o

LDLIBS := -lnvscibuf
LDLIBS += -lnvscisync
LDLIBS += -lnvmedia_iep_sci

ifeq ($(NV_PLATFORM_OS), QNX)
LDLIBS += -lnvdvms_client
endif

ifeq ($(NV_PLATFORM_OS), Linux)
    LDLIBS  += -lpthread
endif

$(TARGETS): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean clobber:
	rm -rf $(OBJS) $(TARGETS)
