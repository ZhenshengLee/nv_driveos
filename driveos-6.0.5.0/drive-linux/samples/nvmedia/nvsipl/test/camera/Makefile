# Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include ../../../../../make/nvdefs.mk

TARGETS = nvsipl_camera

CXXFLAGS := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS) -std=c++14 -fexceptions -frtti -fPIC
CPPFLAGS := -I../../../../../include/nvmedia_6x -I../plugin $(NV_PLATFORM_CPPFLAGS) $(NV_PLATFORM_SDK_INC) -DWIN_INTERFACE_CUSTOM -D_POSIX_C_SOURCE=200112L

LDFLAGS  := $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_TARGET_LIB) $(NV_PLATFORM_LDFLAGS)

OBJS   := CUtils.o
OBJS   += main.o
ifeq ($(NV_PLATFORM_SAFETY), 0)
  OBJS   += CComposite.o
  OBJS   += CCompositeNvSci.o
endif

LDLIBS  += -L../../lib
LDLIBS  += -lnvsipl_sampleplugin
LDLIBS  += -lnvsipl_query
LDLIBS  += -lnvsipl
LDLIBS  += -L../utils/lib/
LDLIBS  += -lnvscibuf
LDLIBS  += -lnvscisync
LDLIBS  += -lnvscistream

ifeq ($(NV_PLATFORM_OS), Linux)
  LDLIBS += -lpthread
  LDLIBS += -ldl
  LDLIBS += -lrt
  LDLIBS += -lstdc++
else
  LDLIBS += -lsocket
  LDLIBS += -lc++
endif

ifeq ($(NV_PLATFORM_OS), QNX)
  CPPFLAGS += -DNVMEDIA_QNX
  LDLIBS += -lnvtime
  LDLIBS += -lnvos_s3_safety
  LDLIBS += -lnvdvms_client
endif

ifeq ($(NV_PLATFORM_SAFETY), 0)
  OBJS += CNvWfd.o
  LDLIBS += -ltegrawfd
  LDLIBS += -lnvmedia2d
endif

.PHONY: default
default: $(TARGETS)

$(TARGETS): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

nvmTest_utils_lib:
	$(MAKE) -C ../utils/lib

clean clobber:
	rm -rf $(OBJS) $(TARGETS)
