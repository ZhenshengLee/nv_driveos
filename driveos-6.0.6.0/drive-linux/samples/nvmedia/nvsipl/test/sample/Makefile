# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include ../../../../../make/nvdefs.mk

TARGETS = nvsipl_sample

CPPFLAGS := $(NV_PLATFORM_CPPFLAGS) $(NV_PLATFORM_SDK_INC)
CPPFLAGS += -I./platform -I../../../../../include/nvmedia_6x

CPPFLAGS += -std=c++14 -fexceptions -frtti -fPIC

LDFLAGS := $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_TARGET_LIB) $(NV_PLATFORM_LDFLAGS)

OBJS := CImageManager.o
OBJS += CUtils.o
OBJS += main.o
ifeq ($(NV_PLATFORM_OS), QNX)
  OBJS += CWFDConsumer.o
  OBJS += CComposite.o
endif

LDLIBS += -lnvsipl
LDLIBS += -lnvscibuf
LDLIBS += -lnvscisync
ifeq ($(NV_PLATFORM_OS), QNX)
  LDLIBS += -lnvmedia2d
  LDLIBS += -ltegrawfd
  LDLIBS += -lnvdtcommon
endif

ifeq ($(NV_PLATFORM_OS), Linux)
  LDLIBS += -lpthread
endif

ifeq ($(NV_PLATFORM_OS), QNX)
  CPPFLAGS += -DNVMEDIA_QNX
endif

.PHONY: default
default: $(TARGETS)

$(TARGETS): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean clobber:
	rm -rf $(OBJS) $(TARGETS)
