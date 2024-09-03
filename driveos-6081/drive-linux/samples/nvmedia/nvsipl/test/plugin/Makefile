# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include ../../../../../make/nvdefs.mk

TARGETS = libnvsipl_sampleplugin.so

CFLAGS   := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS) -I.
CXXFLAGS := $(NV_PLATFORM_SDK_INC)
CXXFLAGS += $(NV_PLATFORM_CPPFLAGS)
CXXFLAGS += $(NV_PLATFORM_SDK_INC) -DWIN_INTERFACE_CUSTOM -D_POSIX_C_SOURCE=200112L -I. -I../../include -I../../query/include -I../utils/include/
CXXFLAGS += -fpic

LDFLAGS  := -L$(NV_PLATFORM_SDK_LIB_DIR) -Wl,--no-undefined

OBJS   := CSamplePlugin.o

ifeq ($(NV_PLATFORM_OS), Linux)
  LDLIBS += -lpthread
  LDLIBS += -ldl
  LDLIBS += -lrt
  LDLIBS += -lstdc++
endif

ifeq ($(NV_PLATFORM_OS), QNX)
  LDLIBS   += -lc++
  CFLAGS   += -DNVMEDIA_QNX
  CXXFLAGS += -DNVMEDIA_QNX
endif

.PHONY: default
default: $(TARGETS)

$(TARGETS): $(OBJS)
	$(CXX) -shared $(LDFLAGS) -o $@ $^ $(LDLIBS) $(LDFLAGS)


clean clobber:
	rm -rf $(OBJS) $(TARGETS)
